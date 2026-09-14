from __future__ import annotations

import pytest

from app.prompt_loader import PROMPTS, PromptConfigurationError, load_prompt_catalog


def test_packaged_prompt_catalog_contains_every_runtime_template() -> None:
    assert PROMPTS.version == 1
    assert "Catholic study companion" in PROMPTS.answer_instructions
    assert "grounding checks" in PROMPTS.repair_instructions

    rendered = PROMPTS.templates.repair_input.format(
        original_input="ORIGINAL",
        issues="- FAILURE",
        rejected_answer="REJECTED",
    )
    assert "ORIGINAL" in rendered
    assert "FAILURE" in rendered
    assert "REJECTED" in rendered


def test_prompt_catalog_rejects_missing_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text(
        """
version: 1
system:
  answer: Answer instructions
  repair: Repair instructions
templates:
  search_query: Search
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptConfigurationError, match="templates.task"):
        load_prompt_catalog(prompt_file)


def test_prompt_catalog_rejects_unknown_schema_version(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("version: 99", encoding="utf-8")

    with pytest.raises(PromptConfigurationError, match="expected 1"):
        load_prompt_catalog(prompt_file)
