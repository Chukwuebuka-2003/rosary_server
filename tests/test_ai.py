from __future__ import annotations

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from app.ai import AiService, InstallationRateLimiter
from app.llm import LlmSettings, create_llm_provider
from app.main import create_app
from app.retrieval import ExaRetriever, ExaSettings, RetrievedSource


INSTALLATION_ID = "671da975-8bf4-4c6b-a79c-7ad886952f73"


class FixedRetriever:
    name = "exa"

    async def search(self, _query: str, _allowed_domains: tuple[str, ...]):
        return (
            RetrievedSource(
                source_id="S1",
                title="The Holy See",
                url="https://www.vatican.va/example",
                excerpt="The shepherd rejoices when he finds the lost sheep.",
            ),
        )


def request_payload() -> dict[str, object]:
    return {
        "task": "explain_reading",
        "question": "Explain today's Gospel.",
        "context": {
            "day": {
                "date": "2026-09-13",
                "calendar": "General Roman Calendar",
                "celebration": "Twenty-fourth Sunday in Ordinary Time",
                "readingReferences": ["Luke 15:1-32"],
            }
        },
        "history": [],
    }


def configured_app(
    tmp_path,
    handler,
    limit: int = 6,
    adapter: str = "responses",
    retriever=None,
):
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = LlmSettings(
        adapter=adapter,
        base_url="https://llm.test/v1",
        api_key="test-key",
        model="test-model",
        chat_token_field="max_tokens",
    )
    service = AiService(create_llm_provider(settings, async_client), retriever)
    limiter = InstallationRateLimiter(limit=limit)
    return create_app(tmp_path, ai_service=service, ai_rate_limiter=limiter), async_client


def test_responses_adapter_uses_server_key_and_returns_citations(tmp_path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["store"] is False
        assert "tools" not in payload
        assert "[S1] The Holy See" in payload["input"]
        assert len(payload["safety_identifier"]) == 64
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Jesus reveals God's joy in seeking the lost [S1].",
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    app, async_client = configured_app(tmp_path, upstream, retriever=FixedRetriever())
    with TestClient(app) as client:
        response = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json=request_payload(),
        )

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["responseId"] == "resp_test"
    assert response.json()["citations"][0]["url"] == "https://www.vatican.va/example"
    assert response.json()["limitations"] == []
    asyncio.run(async_client.aclose())


def test_ai_rejects_invalid_installation_and_payload(tmp_path) -> None:
    app = create_app(tmp_path)
    with TestClient(app) as client:
        invalid_id = client.post("/v1/ai/ask", json=request_payload())
        invalid_question = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json={**request_payload(), "question": " "},
        )

    assert invalid_id.status_code == 400
    assert invalid_id.json()["error"] == "invalid_installation_id"
    assert invalid_question.status_code == 400
    assert invalid_question.json()["error"] == "invalid_request"


def test_ai_reports_missing_server_configuration(tmp_path, monkeypatch) -> None:
    for variable in (
        "LLM_ADAPTER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "EXA_API_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    app = create_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health/ai")
        response = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json=request_payload(),
        )

    assert health.json() == {"status": "not_configured", "search": "not_configured"}
    assert response.status_code == 503
    assert response.json()["error"] == "ai_not_configured"


def test_ai_rate_limits_by_installation(tmp_path) -> None:
    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Answer", "annotations": []}
                        ],
                    }
                ],
            },
        )

    app, async_client = configured_app(tmp_path, upstream, limit=1)
    with TestClient(app) as client:
        first = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json=request_payload(),
        )
        second = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json=request_payload(),
        )

    assert first.status_code == 200
    assert first.json()["limitations"]
    assert second.status_code == 429
    asyncio.run(async_client.aclose())


def test_ai_discards_unknown_source_markers(tmp_path) -> None:
    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "An unsupported answer [S9].",
                                "annotations": [],
                            }
                        ],
                    }
                ],
            },
        )

    app, async_client = configured_app(tmp_path, upstream, retriever=FixedRetriever())
    with TestClient(app) as client:
        response = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json=request_payload(),
        )

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["limitations"]
    asyncio.run(async_client.aclose())


def test_chat_completions_adapter_is_swappable_without_android_changes(tmp_path) -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["max_tokens"] == 700
        return httpx.Response(
            200,
            json={
                "id": "chat_test",
                "choices": [
                    {"message": {"role": "assistant", "content": "A provider-neutral answer."}}
                ],
            },
        )

    app, async_client = configured_app(
        tmp_path,
        upstream,
        adapter="chat_completions",
    )
    with TestClient(app) as client:
        health = client.get("/health/ai")
        response = client.post(
            "/v1/ai/ask",
            headers={"X-Installation-Id": INSTALLATION_ID},
            json=request_payload(),
        )

    assert health.json() == {
        "status": "configured",
        "adapter": "chat_completions",
        "search": "not_configured",
    }
    assert response.status_code == 200
    assert response.json()["answer"] == "A provider-neutral answer."
    assert response.json()["limitations"]
    asyncio.run(async_client.aclose())


def test_exa_search_uses_domain_filter_and_highlights() -> None:
    def exa(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.exa.ai/search"
        assert request.headers["x-api-key"] == "exa-test-key"
        payload = json.loads(request.content)
        assert payload["type"] == "fast"
        assert payload["includeDomains"] == ["vatican.va", "usccb.org"]
        assert payload["contents"]["highlights"]["maxCharacters"] == 1_500
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Catechism of the Catholic Church",
                        "url": "https://www.vatican.va/archive/ENG0015/_INDEX.HTM",
                        "highlights": ["The Gospel is the revelation in Jesus Christ."],
                    },
                    {
                        "title": "Wrong domain",
                        "url": "https://example.com/page",
                        "highlights": ["This must not enter the prompt."],
                    },
                ]
            },
        )

    async def run_search():
        client = httpx.AsyncClient(transport=httpx.MockTransport(exa))
        try:
            retriever = ExaRetriever(
                client,
                ExaSettings(
                    api_key="exa-test-key",
                    base_url="https://api.exa.ai",
                    num_results=5,
                ),
            )
            return await retriever.search("Catholic Gospel", ("vatican.va", "usccb.org"))
        finally:
            await client.aclose()

    sources = asyncio.run(run_search())
    assert len(sources) == 1
    assert sources[0].source_id == "S1"
    assert sources[0].title == "Catechism of the Catholic Church"
