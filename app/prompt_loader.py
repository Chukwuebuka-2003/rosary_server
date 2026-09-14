from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml


PROMPT_SCHEMA_VERSION = 1


class PromptConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptTemplates:
    search_query: str
    task: str
    context: str
    history_item: str
    history: str
    source: str
    sources: str
    no_sources: str
    question: str
    repair_issue: str
    repair_input: str
    grounding_fallback_answer: str
    grounding_fallback_limitation: str


@dataclass(frozen=True)
class PromptCatalog:
    version: int
    answer_instructions: str
    repair_instructions: str
    templates: PromptTemplates


def load_prompt_catalog(path: Path | None = None) -> PromptCatalog:
    prompt_path = path or files("app").joinpath("prompts.yaml")
    try:
        value = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise PromptConfigurationError(f"Unable to load prompt configuration: {error}") from error

    if not isinstance(value, dict):
        raise PromptConfigurationError("Prompt configuration must be a YAML mapping.")
    version = value.get("version")
    if version != PROMPT_SCHEMA_VERSION:
        raise PromptConfigurationError(
            f"Unsupported prompt schema version {version!r}; expected {PROMPT_SCHEMA_VERSION}."
        )

    system = require_mapping(value, "system")
    templates = require_mapping(value, "templates")
    return PromptCatalog(
        version=version,
        answer_instructions=require_prompt(system, "answer", "system.answer"),
        repair_instructions=require_prompt(system, "repair", "system.repair"),
        templates=PromptTemplates(
            search_query=require_prompt(templates, "search_query", "templates.search_query"),
            task=require_prompt(templates, "task", "templates.task"),
            context=require_prompt(templates, "context", "templates.context"),
            history_item=require_prompt(templates, "history_item", "templates.history_item"),
            history=require_prompt(templates, "history", "templates.history"),
            source=require_prompt(templates, "source", "templates.source"),
            sources=require_prompt(templates, "sources", "templates.sources"),
            no_sources=require_prompt(templates, "no_sources", "templates.no_sources"),
            question=require_prompt(templates, "question", "templates.question"),
            repair_issue=require_prompt(templates, "repair_issue", "templates.repair_issue"),
            repair_input=require_prompt(templates, "repair_input", "templates.repair_input"),
            grounding_fallback_answer=require_prompt(
                templates,
                "grounding_fallback_answer",
                "templates.grounding_fallback_answer",
            ),
            grounding_fallback_limitation=require_prompt(
                templates,
                "grounding_fallback_limitation",
                "templates.grounding_fallback_limitation",
            ),
        ),
    )


def require_mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise PromptConfigurationError(f"Prompt configuration entry {key!r} must be a mapping.")
    return nested


def require_prompt(value: dict[str, Any], key: str, label: str) -> str:
    prompt = value.get(key)
    if not isinstance(prompt, str) or not prompt.strip():
        raise PromptConfigurationError(f"Prompt configuration entry {label!r} must be non-empty text.")
    return prompt.strip()


PROMPTS = load_prompt_catalog()
