from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from typing import Any, Literal, Protocol, cast
from urllib.parse import urlparse

import httpx


MAX_RESPONSE_BYTES = 262_144


@dataclass(frozen=True)
class LlmPrompt:
    instructions: str
    input: str
    end_user_id: str
    max_output_tokens: int


@dataclass(frozen=True)
class LlmResult:
    response_id: str
    text: str


class LanguageModelProvider(Protocol):
    """The only interface the Catholic answer service depends on."""

    name: str

    async def generate(self, prompt: LlmPrompt) -> LlmResult: ...


class LlmProviderError(Exception):
    def __init__(self, kind: Literal["busy", "unavailable", "invalid_response"]) -> None:
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class LlmSettings:
    adapter: str | None
    base_url: str | None
    api_key: str | None
    model: str | None
    chat_token_field: Literal["max_tokens", "max_completion_tokens"]

    @classmethod
    def from_environment(cls) -> LlmSettings:
        token_field = os.environ.get("LLM_CHAT_TOKEN_FIELD", "max_tokens").strip().lower()
        if token_field not in {"max_tokens", "max_completion_tokens"}:
            token_field = "max_tokens"
        return cls(
            adapter=os.environ.get("LLM_ADAPTER", "").strip().lower() or None,
            base_url=os.environ.get("LLM_BASE_URL", "").strip().rstrip("/") or None,
            api_key=os.environ.get("LLM_API_KEY", "").strip() or None,
            model=os.environ.get("LLM_MODEL", "").strip() or None,
            chat_token_field=cast(
                Literal["max_tokens", "max_completion_tokens"],
                token_field,
            ),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.adapter and self.base_url and self.model)


def create_llm_provider(
    settings: LlmSettings,
    client: httpx.AsyncClient,
) -> LanguageModelProvider | None:
    if not settings.is_configured:
        return None
    assert settings.adapter and settings.base_url and settings.model
    validate_base_url(settings.base_url)
    common = {
        "client": client,
        "base_url": settings.base_url,
        "api_key": settings.api_key,
        "model": settings.model,
    }
    if settings.adapter == "responses":
        return ResponsesApiProvider(**common)
    if settings.adapter == "chat_completions":
        return ChatCompletionsProvider(
            **common,
            token_field=settings.chat_token_field,
        )
    raise ValueError(
        f"Unsupported LLM_ADAPTER {settings.adapter!r}; use 'responses' or 'chat_completions'."
    )


class HttpLlmProvider:
    name = "http"

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        api_key: str | None,
        model: str,
    ) -> None:
        self.client = client
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            response = await self.client.post(
                f"{self.base_url}{path}",
                headers=self.headers(),
                json=body,
            )
        except (httpx.TimeoutException, httpx.NetworkError):
            raise LlmProviderError("unavailable") from None
        if response.status_code == 429:
            raise LlmProviderError("busy")
        if response.status_code < 200 or response.status_code >= 300:
            raise LlmProviderError("unavailable")
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise LlmProviderError("invalid_response")
        try:
            return response.json()
        except ValueError:
            raise LlmProviderError("invalid_response") from None


class ResponsesApiProvider(HttpLlmProvider):
    name = "responses"

    async def generate(self, prompt: LlmPrompt) -> LlmResult:
        body: dict[str, Any] = {
            "model": self.model,
            "store": False,
            "instructions": prompt.instructions,
            "input": prompt.input,
            "max_output_tokens": prompt.max_output_tokens,
            "safety_identifier": sha256(prompt.end_user_id.encode()).hexdigest(),
        }
        value = await self.post("/responses", body)
        try:
            return parse_responses_result(value)
        except (ValueError, TypeError, KeyError):
            raise LlmProviderError("invalid_response") from None


class ChatCompletionsProvider(HttpLlmProvider):
    name = "chat_completions"

    def __init__(
        self,
        *args: Any,
        token_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.token_field = token_field

    async def generate(self, prompt: LlmPrompt) -> LlmResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.instructions},
                {"role": "user", "content": prompt.input},
            ],
            self.token_field: prompt.max_output_tokens,
            "stream": False,
        }
        value = await self.post("/chat/completions", body)
        try:
            response_id = value["id"]
            content = value["choices"][0]["message"]["content"]
            text = chat_content_text(content)
            if not isinstance(response_id, str) or not text.strip():
                raise ValueError("invalid chat completion")
            return LlmResult(response_id=response_id, text=text.strip())
        except (IndexError, KeyError, TypeError, ValueError):
            raise LlmProviderError("invalid_response") from None


def chat_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    return ""


def parse_responses_result(value: Any) -> LlmResult:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str):
        raise ValueError("invalid response shape")
    output = value.get("output")
    if not isinstance(output, list):
        raise ValueError("missing output")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content_items = item.get("content")
        if not isinstance(content_items, list):
            continue
        for content in content_items:
            if (
                not isinstance(content, dict)
                or content.get("type") != "output_text"
                or not isinstance(content.get("text"), str)
                or not content["text"].strip()
            ):
                continue
            return LlmResult(
                response_id=value["id"],
                text=content["text"].strip(),
            )
    raise ValueError("missing output text")


def validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme == "https" and parsed.netloc:
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError("LLM_BASE_URL must use HTTPS, except for a local loopback model server.")
