from __future__ import annotations

from collections import OrderedDict, deque
import json
import logging
import os
import re
from threading import Lock
from time import monotonic
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.llm import LanguageModelProvider, LlmPrompt, LlmProviderError, LlmResult
from app.prompt_loader import PROMPTS
from app.retrieval import RetrievedSource, RetrievalError, SourceRetriever


MAX_REQUEST_BYTES = 16_384
INSTALLATION_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
ALLOWED_SOURCE_DOMAINS = ("vatican.va", "usccb.org")
SOURCE_MARKER_PATTERN = re.compile(r"\[(S[1-9][0-9]*)\]")
SOURCE_MARKER_GROUP_PATTERN = re.compile(
    r"[\[【]\s*(S[1-9][0-9]*(?:\s*[,;]\s*S[1-9][0-9]*)*)\s*[\]】]"
)
QUOTATION_PATTERN = re.compile(r"(?:[\"“][^\"”\n]{20,}[\"”])")
REFERENCE_RULES = (
    (
        re.compile(
            r"\b(?:CCC|Catechism(?: of the Catholic Church)?)\s*(?:§{1,2}\s*)?\d{1,4}\b",
            re.IGNORECASE,
        ),
        ("catechism", "eng0015"),
        "Catechism",
    ),
    (
        re.compile(r"\b(?:canon|can\.)\s*\d{1,4}\b", re.IGNORECASE),
        ("code of canon law", "cod-iuris-canonici"),
        "Canon Law",
    ),
)
CITATION_COVERAGE_THRESHOLD = 0.75
LOGGER = logging.getLogger(__name__)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryItem(StrictModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=1_200)


class AiAskRequest(StrictModel):
    task: Literal["explain_reading", "explain_mystery", "follow_up"]
    question: str = Field(min_length=2, max_length=800)
    context: dict[str, Any]
    history: list[HistoryItem] = Field(default_factory=list, max_length=6)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 2:
            raise ValueError("question must contain at least two non-space characters")
        return cleaned


class AiCitation(StrictModel):
    title: str
    url: str
    startIndex: int = Field(ge=0)
    endIndex: int = Field(ge=0)


class AiAnswer(StrictModel):
    answer: str
    citations: list[AiCitation]
    limitations: list[str]
    responseId: str


class AiServiceError(Exception):
    def __init__(self, error: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.status_code = status_code


class InstallationRateLimiter:
    """Small single-process limiter; use a shared edge/Redis limiter when scaling out."""

    def __init__(self, limit: int, window_seconds: float = 60.0, max_keys: int = 10_000) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = monotonic() if now is None else now
        cutoff = timestamp - self.window_seconds
        with self._lock:
            timestamps = self._requests.pop(key, deque())
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            allowed = len(timestamps) < self.limit
            if allowed:
                timestamps.append(timestamp)
            self._requests[key] = timestamps
            while len(self._requests) > self.max_keys:
                self._requests.popitem(last=False)
            return allowed


class AiService:
    """Provider-neutral Catholic answer orchestration used by the HTTP route."""

    def __init__(
        self,
        provider: LanguageModelProvider | None,
        retriever: SourceRetriever | None = None,
    ) -> None:
        self.provider = provider
        self.retriever = retriever

    @property
    def is_configured(self) -> bool:
        return self.provider is not None

    @property
    def provider_name(self) -> str | None:
        return self.provider.name if self.provider else None

    @property
    def retrieval_name(self) -> str | None:
        return self.retriever.name if self.retriever else None

    async def ask(self, request: AiAskRequest, installation_id: str) -> AiAnswer:
        if self.provider is None:
            raise AiServiceError(
                "ai_not_configured",
                "The explanation service is not configured.",
                503,
            )
        sources: tuple[RetrievedSource, ...] = ()
        if self.retriever is not None:
            try:
                sources = await self.retriever.search(
                    build_search_query(request),
                    ALLOWED_SOURCE_DOMAINS,
                )
            except RetrievalError:
                raise AiServiceError(
                    "search_unavailable",
                    "The official-source search service is temporarily unavailable.",
                    503,
                ) from None

        prompt_input = build_prompt_input(request, sources)
        result = await self._generate(
            LlmPrompt(
                instructions=PROMPTS.answer_instructions,
                input=prompt_input,
                end_user_id=installation_id,
                max_output_tokens=700,
            )
        )
        result = canonicalize_result_markers(result)
        issues = grounding_issues(result.text, sources)
        if issues:
            result = await self._generate(
                LlmPrompt(
                    instructions=(
                        f"{PROMPTS.answer_instructions}\n\n{PROMPTS.repair_instructions}"
                    ),
                    input=build_repair_input(prompt_input, result.text, issues),
                    end_user_id=installation_id,
                    max_output_tokens=700,
                )
            )
            result = canonicalize_result_markers(result)
            issues = grounding_issues(result.text, sources)
        if issues:
            # A rejected draft must never reach the client. Grounding failure is a
            # supported product outcome, though, rather than an upstream outage.
            # Return a deterministic, claim-free limitation so the app can recover
            # without presenting generated text as verified Catholic teaching.
            LOGGER.warning(
                "AI answer failed grounding after repair; response_id=%s issues=%s",
                result.response_id,
                issues,
            )
            return AiAnswer(
                answer=PROMPTS.templates.grounding_fallback_answer,
                citations=[],
                limitations=[PROMPTS.templates.grounding_fallback_limitation],
                responseId=result.response_id,
            )

        citations = citations_from_answer(result.text, sources)
        limitations = (
            []
            if citations
            else ["No supporting source citation was returned. Treat this answer as unverified."]
        )
        return AiAnswer(
            answer=result.text,
            citations=citations,
            limitations=limitations,
            responseId=result.response_id,
        )

    async def _generate(self, prompt: LlmPrompt) -> LlmResult:
        assert self.provider is not None
        try:
            return await self.provider.generate(prompt)
        except LlmProviderError as error:
            if error.kind == "busy":
                raise AiServiceError(
                    "ai_unavailable",
                    "The explanation service is busy. Please try again shortly.",
                    503,
                ) from None
            if error.kind == "invalid_response":
                raise AiServiceError(
                    "invalid_ai_response",
                    "The explanation service returned an invalid response.",
                    502,
                ) from None
            raise AiServiceError(
                "ai_unavailable",
                "The explanation service is temporarily unavailable.",
                503,
            ) from None


def requests_per_minute_from_environment() -> int:
    raw_limit = os.environ.get("AI_REQUESTS_PER_MINUTE", "6")
    try:
        return max(1, min(int(raw_limit), 60))
    except ValueError:
        return 6


def parse_ask_request(body: bytes) -> AiAskRequest:
    try:
        return AiAskRequest.model_validate_json(body)
    except ValidationError as error:
        first = error.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first["loc"])
        message = f"{location}: {first['msg']}" if location else first["msg"]
        raise AiServiceError("invalid_request", message, 400) from None


def build_search_query(request: AiAskRequest) -> str:
    context = json.dumps(request.context, ensure_ascii=True, separators=(",", ":"))
    return PROMPTS.templates.search_query.format(
        task=request.task,
        question=request.question,
        context=context,
    )[:1_500]


def build_prompt_input(
    request: AiAskRequest,
    sources: tuple[RetrievedSource, ...] = (),
) -> str:
    templates = PROMPTS.templates
    history = "\n".join(
        templates.history_item.format(role=item.role.upper(), text=item.text)
        for item in request.history
    )
    context = json.dumps(request.context, separators=(",", ":"))
    input_parts = [
        templates.task.format(task=request.task),
        templates.context.format(context=context),
    ]
    if history:
        input_parts.append(templates.history.format(history=history))
    if sources:
        source_text = "\n\n".join(
            templates.source.format(
                source_id=source.source_id,
                title=source.title,
                url=source.url,
                excerpt=source.excerpt,
            )
            for source in sources
        )
        input_parts.append(templates.sources.format(sources=source_text))
    else:
        input_parts.append(templates.no_sources)
    input_parts.append(templates.question.format(question=request.question))
    return "\n\n".join(input_parts)


def build_repair_input(original_input: str, rejected_answer: str, issues: list[str]) -> str:
    issue_text = "\n".join(
        PROMPTS.templates.repair_issue.format(issue=issue) for issue in issues
    )
    return PROMPTS.templates.repair_input.format(
        original_input=original_input,
        issues=issue_text,
        rejected_answer=rejected_answer,
    )


def grounding_issues(answer: str, sources: tuple[RetrievedSource, ...]) -> list[str]:
    source_by_id = {source.source_id: source for source in sources}
    marker_ids = SOURCE_MARKER_PATTERN.findall(answer)
    issues: list[str] = []

    unknown_ids = sorted(set(marker_ids) - source_by_id.keys())
    if unknown_ids:
        issues.append(f"Unknown source markers: {', '.join(unknown_ids)}.")

    valid_marker_ids = set(marker_ids) & source_by_id.keys()
    if sources and not valid_marker_ids:
        issues.append("No supplied official source was cited.")

    paragraphs = answer_paragraphs(answer)
    substantive = [part for part in paragraphs if is_substantive_paragraph(part)]
    if sources and substantive:
        cited_count = sum(paragraph_ends_with_valid_marker(part, source_by_id) for part in substantive)
        coverage = cited_count / len(substantive)
        if coverage < CITATION_COVERAGE_THRESHOLD:
            issues.append(
                f"Only {cited_count} of {len(substantive)} substantive paragraphs end with supplied sources."
            )

    for paragraph in paragraphs:
        paragraph_source_ids = valid_markers_in(paragraph, source_by_id)
        for quotation_match in QUOTATION_PATTERN.finditer(paragraph):
            if not paragraph_source_ids:
                issues.append("A quotation is not cited in the same paragraph.")
                continue
            quotation = quotation_match.group(0)[1:-1]
            if not any(
                normalized_text(quotation) in normalized_text(source_by_id[source_id].excerpt)
                for source_id in paragraph_source_ids
            ):
                issues.append("A quotation is not present in the cited source excerpt.")

        for pattern, source_hints, label in REFERENCE_RULES:
            reference_match = pattern.search(paragraph)
            if reference_match is None:
                continue
            if not paragraph_source_ids:
                issues.append(f"An {label} reference is not cited in the same paragraph.")
                continue
            referenced_sources = (source_by_id[source_id] for source_id in paragraph_source_ids)
            if not any(
                source_supports_reference(source, source_hints, reference_match.group(0))
                for source in referenced_sources
            ):
                issues.append(
                    f"An {label} reference cites a source that does not contain that exact reference."
                )

    return list(dict.fromkeys(issues))


def canonicalize_result_markers(result: LlmResult) -> LlmResult:
    """Normalize equivalent LLM citation syntax before strict source validation."""

    def replace_group(match: re.Match[str]) -> str:
        source_ids = re.findall(r"S[1-9][0-9]*", match.group(1))
        return " ".join(f"[{source_id}]" for source_id in source_ids)

    normalized = SOURCE_MARKER_GROUP_PATTERN.sub(replace_group, result.text)
    return LlmResult(response_id=result.response_id, text=normalized)


def valid_markers_in(paragraph: str, source_by_id: dict[str, RetrievedSource]) -> set[str]:
    return set(SOURCE_MARKER_PATTERN.findall(paragraph)) & source_by_id.keys()


def answer_paragraphs(answer: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"\n\s*\n|\n(?=\s*(?:[-*+]\s+|\d+[.)]\s+))", answer)
        if part.strip()
    ]


def paragraph_ends_with_valid_marker(
    paragraph: str,
    source_by_id: dict[str, RetrievedSource],
) -> bool:
    matches = list(SOURCE_MARKER_PATTERN.finditer(paragraph))
    if not matches or matches[-1].group(1) not in source_by_id:
        return False
    suffix = paragraph[matches[-1].end() :]
    return re.fullmatch(r"[\s.,;:!?)*_`]*", suffix) is not None


def source_supports_reference(
    source: RetrievedSource,
    hints: tuple[str, ...],
    reference: str,
) -> bool:
    searchable = f"{source.title} {source.url} {source.excerpt}".lower()
    if not any(hint in searchable for hint in hints):
        return False
    reference_numbers = re.findall(r"\d+", reference)
    return all(re.search(rf"\b{re.escape(number)}\b", searchable) for number in reference_numbers)


def normalized_text(value: str) -> str:
    return " ".join(
        value.casefold()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .split()
    )


def is_substantive_paragraph(paragraph: str) -> bool:
    plain = SOURCE_MARKER_PATTERN.sub("", paragraph)
    plain = re.sub(r"[*_#>`]", "", plain).strip()
    lowered = re.sub(r"^[\s\-\d.)]+", "", plain).lower()
    if lowered.startswith(("reflection (generated)", "devotional reflection (generated)")):
        return False
    words = re.findall(r"\b[\w’'-]+\b", plain)
    if len(words) < 8:
        return False
    if plain.endswith(":") and len(words) <= 14:
        return False
    return True


def citations_from_answer(
    answer: str,
    sources: tuple[RetrievedSource, ...],
) -> list[AiCitation]:
    source_by_id = {source.source_id: source for source in sources}
    cited: set[str] = set()
    citations: list[AiCitation] = []
    for match in SOURCE_MARKER_PATTERN.finditer(answer):
        source_id = match.group(1)
        source = source_by_id.get(source_id)
        if source is None or source_id in cited:
            continue
        cited.add(source_id)
        citations.append(
            AiCitation(
                title=source.title,
                url=source.url,
                startIndex=match.start(),
                endIndex=match.end(),
            )
        )
    return citations
