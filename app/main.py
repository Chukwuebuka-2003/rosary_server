from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
import os
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.ai import (
    INSTALLATION_ID_PATTERN,
    MAX_REQUEST_BYTES,
    AiService,
    AiServiceError,
    InstallationRateLimiter,
    parse_ask_request,
    requests_per_minute_from_environment,
)
from app.llm import LlmSettings, create_llm_provider
from app.models import (
    CalendarBundle,
    CalendarScope,
    LiturgicalDayResponse,
    ReleaseManifestItem,
)
from app.store import ReleaseNotFoundError, ReleaseStore
from app.retrieval import ExaRetriever, ExaSettings


def default_content_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def create_app(
    content_root: Path | None = None,
    ai_service: AiService | None = None,
    ai_rate_limiter: InstallationRateLimiter | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(running_app: FastAPI):
        if running_app.state.ai_service is None:
            running_app.state.ai_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            provider = create_llm_provider(
                running_app.state.llm_settings,
                running_app.state.ai_http_client,
            )
            retriever = (
                ExaRetriever(running_app.state.ai_http_client, running_app.state.exa_settings)
                if running_app.state.exa_settings.api_key
                else None
            )
            running_app.state.ai_service = AiService(
                provider,
                retriever,
            )
        try:
            yield
        finally:
            if running_app.state.ai_http_client is not None:
                await running_app.state.ai_http_client.aclose()

    app = FastAPI(
        title="Catholic Companion API",
        version="0.1.0",
        description="Versioned, reviewed content for the Catholic Companion Android app.",
        lifespan=lifespan,
    )
    app.state.release_store = ReleaseStore(
        content_root or Path(os.environ.get("CATHOLIC_CONTENT_DIR", default_content_root()))
    )
    app.state.ai_http_client = None
    app.state.ai_service = ai_service
    app.state.llm_settings = LlmSettings.from_environment()
    app.state.exa_settings = ExaSettings.from_environment()
    app.state.ai_rate_limiter = ai_rate_limiter or InstallationRateLimiter(
        requests_per_minute_from_environment()
    )

    def get_store(request: Request) -> ReleaseStore:
        return request.app.state.release_store

    @app.exception_handler(ReleaseNotFoundError)
    async def not_found_handler(_request: Request, error: ReleaseNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ai")
    def ai_health(request: Request) -> dict[str, str]:
        service: AiService = request.app.state.ai_service
        result = {"status": "configured" if service.is_configured else "not_configured"}
        if service.provider_name:
            result["adapter"] = service.provider_name
        result["search"] = service.retrieval_name or "not_configured"
        return result

    @app.post("/v1/ai/ask")
    async def ask_ai(request: Request) -> JSONResponse:
        installation_id = request.headers.get("X-Installation-Id", "")
        if not INSTALLATION_ID_PATTERN.fullmatch(installation_id):
            return ai_json({"error": "invalid_installation_id"}, 400)

        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return ai_json({"error": "request_too_large"}, 413)
            except ValueError:
                return ai_json({"error": "invalid_content_length"}, 400)

        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            return ai_json({"error": "request_too_large"}, 413)
        try:
            payload = parse_ask_request(body)
        except AiServiceError as error:
            return ai_json({"error": error.error, "message": error.message}, error.status_code)

        if not request.app.state.ai_rate_limiter.allow(installation_id):
            return ai_json(
                {"error": "rate_limited", "message": "Please wait before asking again."},
                429,
            )

        try:
            answer = await request.app.state.ai_service.ask(payload, installation_id)
        except AiServiceError as error:
            return ai_json({"error": error.error, "message": error.message}, error.status_code)
        return ai_json(answer.model_dump(), 200)

    @app.get("/v1/calendar-scopes", response_model=list[CalendarScope], response_model_by_alias=True)
    def calendar_scopes(store: ReleaseStore = Depends(get_store)) -> list[CalendarScope]:
        latest: dict[str, CalendarScope] = {}
        for release in store.published_releases():
            latest[release.bundle.scope.id] = release.bundle.scope
        return list(latest.values())

    @app.get("/v1/content/manifest", response_model=list[ReleaseManifestItem], response_model_by_alias=True)
    def manifest(store: ReleaseStore = Depends(get_store)) -> list[ReleaseManifestItem]:
        return [
            ReleaseManifestItem(
                release_id=release.release_id,
                calendar_id=release.bundle.scope.id,
                coverage_start=release.bundle.scope.coverage_start,
                coverage_end=release.bundle.scope.coverage_end,
                checksum_sha256=release.checksum_sha256,
                download_url=f"/v1/content/releases/{release.release_id}/bundle",
            )
            for release in store.published_releases()
        ]

    @app.get(
        "/v1/content/releases/{release_id}/bundle",
        response_model=CalendarBundle,
        response_model_by_alias=True,
    )
    def bundle(release_id: str, store: ReleaseStore = Depends(get_store)) -> CalendarBundle:
        return store.get_published(release_id).bundle

    @app.get("/v1/liturgy/day", response_model=LiturgicalDayResponse, response_model_by_alias=True)
    def liturgical_day(
        requested_date: date = Query(alias="date"),
        calendar_id: str = Query(default="general-roman"),
        store: ReleaseStore = Depends(get_store),
    ) -> LiturgicalDayResponse:
        release = store.release_for_day(calendar_id, requested_date)
        days = [day for day in release.bundle.days if day.date == requested_date]
        if len(days) != 1:
            raise HTTPException(status_code=500, detail="published release has an invalid day record")
        celebrations = [
            celebration
            for celebration in release.bundle.celebrations
            if celebration.date == requested_date and celebration.calendar_id == calendar_id
        ]
        celebration_ids = {celebration.id for celebration in celebrations}
        readings = [
            reading
            for reading in release.bundle.readings
            if reading.celebration_id in celebration_ids
        ]
        return LiturgicalDayResponse(
            scope=release.bundle.scope,
            day=days[0],
            celebrations=celebrations,
            readings=readings,
        )

    return app


def ai_json(content: dict[str, object], status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


app = create_app()
