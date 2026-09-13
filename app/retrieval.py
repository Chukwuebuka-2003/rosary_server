from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx


MAX_EXA_RESPONSE_BYTES = 524_288
MAX_HIGHLIGHT_CHARACTERS = 1_500


@dataclass(frozen=True)
class RetrievedSource:
    source_id: str
    title: str
    url: str
    excerpt: str


class SourceRetriever(Protocol):
    name: str

    async def search(
        self,
        query: str,
        allowed_domains: tuple[str, ...],
    ) -> tuple[RetrievedSource, ...]: ...


class RetrievalError(Exception):
    pass


@dataclass(frozen=True)
class ExaSettings:
    api_key: str | None
    base_url: str
    num_results: int

    @classmethod
    def from_environment(cls) -> ExaSettings:
        raw_results = os.environ.get("EXA_NUM_RESULTS", "5")
        try:
            num_results = max(1, min(int(raw_results), 10))
        except ValueError:
            num_results = 5
        return cls(
            api_key=os.environ.get("EXA_API_KEY", "").strip() or None,
            base_url=os.environ.get("EXA_BASE_URL", "https://api.exa.ai").strip().rstrip("/"),
            num_results=num_results,
        )


class ExaRetriever:
    name = "exa"

    def __init__(self, client: httpx.AsyncClient, settings: ExaSettings) -> None:
        if not settings.api_key:
            raise ValueError("EXA_API_KEY is required to configure Exa search.")
        validate_exa_base_url(settings.base_url)
        self.client = client
        self.settings = settings

    async def search(
        self,
        query: str,
        allowed_domains: tuple[str, ...],
    ) -> tuple[RetrievedSource, ...]:
        try:
            response = await self.client.post(
                f"{self.settings.base_url}/search",
                headers={
                    "x-api-key": self.settings.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "type": "fast",
                    "numResults": self.settings.num_results,
                    "includeDomains": list(allowed_domains),
                    "moderation": True,
                    "contents": {
                        "highlights": {
                            "query": query,
                            "maxCharacters": MAX_HIGHLIGHT_CHARACTERS,
                        }
                    },
                },
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            raise RetrievalError("Exa search is unavailable.") from None
        if response.status_code < 200 or response.status_code >= 300:
            raise RetrievalError("Exa search failed.")
        if len(response.content) > MAX_EXA_RESPONSE_BYTES:
            raise RetrievalError("Exa search response was too large.")
        try:
            return parse_exa_results(response.json(), allowed_domains)
        except (ValueError, TypeError, KeyError):
            raise RetrievalError("Exa returned an invalid response.") from None


def parse_exa_results(
    value: Any,
    allowed_domains: tuple[str, ...],
) -> tuple[RetrievedSource, ...]:
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError("missing Exa results")
    sources: list[RetrievedSource] = []
    seen_urls: set[str] = set()
    for result in value["results"]:
        if not isinstance(result, dict):
            continue
        title = result.get("title")
        url = result.get("url")
        highlights = result.get("highlights")
        if (
            not isinstance(title, str)
            or not title.strip()
            or not isinstance(url, str)
            or url in seen_urls
            or not is_allowed_source_url(url, allowed_domains)
            or not isinstance(highlights, list)
        ):
            continue
        excerpt = "\n".join(
            highlight.strip()
            for highlight in highlights
            if isinstance(highlight, str) and highlight.strip()
        )[:MAX_HIGHLIGHT_CHARACTERS]
        if not excerpt:
            continue
        seen_urls.add(url)
        sources.append(
            RetrievedSource(
                source_id=f"S{len(sources) + 1}",
                title=title.strip()[:300],
                url=url,
                excerpt=excerpt,
            )
        )
    return tuple(sources)


def is_allowed_source_url(url: str, allowed_domains: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    hostname = parsed.hostname.lower()
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in allowed_domains)


def validate_exa_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("EXA_BASE_URL must be an HTTPS URL.")
