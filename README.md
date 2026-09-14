# Catholic Companion API

FastAPI service for reviewed, versioned liturgical content and the Android app's
secure AI endpoint. The first configured calendar scope is the English **General
Roman Calendar** (`general-roman`). It deliberately excludes Nigerian, Vatican
City, diocesan and parish observances.

## Local setup

```shell
uv sync
uv run pytest
uv run uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

## AI configuration

The Android app calls `POST /v1/ai/ask` on this FastAPI service. The route uses
the provider-neutral `LanguageModelProvider` interface in `app/llm.py`; provider
protocol details and credentials never enter the Android app.

Configure an adapter only in the backend hosting environment:

```shell
export LLM_ADAPTER="responses"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="your-server-side-key"
export LLM_MODEL="gpt-5.5"
export EXA_API_KEY="your-server-side-exa-key"
export AI_REQUESTS_PER_MINUTE="6"
```

That example uses a Responses-style API. For a hosted or self-hosted model with
an OpenAI-compatible Chat Completions endpoint, switch only the environment:

```shell
export LLM_ADAPTER="chat_completions"
export LLM_BASE_URL="https://your-provider.example/v1"
export LLM_API_KEY="your-server-side-key"
export LLM_MODEL="your-model-id"
export LLM_CHAT_TOKEN_FIELD="max_tokens"
export EXA_API_KEY="your-server-side-exa-key"
```

`LLM_API_KEY` is optional for a loopback self-hosted model. Remote endpoints must
use HTTPS; `http://localhost` and `http://127.0.0.1` are allowed for local model
servers. `LLM_CHAT_TOKEN_FIELD` can be `max_tokens` or `max_completion_tokens`.

Adding a native LLM provider later requires one adapter implementing
`LanguageModelProvider`; the Catholic prompts, validation, rate limits, FastAPI
route, Exa retrieval, and Android app do not change.

All LLM instructions, retrieval-query wording, context blocks, source blocks,
and repair templates live in `app/prompts.yaml`. The file has an explicit schema
version and is validated when the application imports it; a missing, empty, or
unsupported prompt configuration stops startup instead of silently changing AI
behaviour. `prompts.yaml` is included as Python package data, so it is available
in editable installs and built distributions.

Exa search is configured independently with `EXA_API_KEY`, optional
`EXA_BASE_URL`, and optional `EXA_NUM_RESULTS` (1–10). FastAPI sends the current
question and displayed liturgical context to Exa, restricts results to
`vatican.va` and `usccb.org`, requests bounded highlights, and rejects returned
URLs outside that allowlist. Retrieved excerpts receive stable source IDs such
as `[S1]`; only IDs actually referenced in the LLM answer become clickable app
citations. Generated answers pass a deterministic grounding check before they
are returned. The check rejects unknown source IDs, uncited quotations,
Catechism or Canon Law references that do not point to a matching retrieved
document, and answers whose substantive paragraphs have less than 75% citation
coverage. A failed draft receives one constrained rewrite attempt; a second
failure returns `ungrounded_ai_response` instead of exposing the draft. If Exa
returns no evidence or the model cites no supplied source, the answer is
explicitly labelled unverified. `GET /health/ai` reports the active LLM adapter
and search provider without revealing secrets.

The included rate limiter is per FastAPI process. Before running multiple API
workers or exposing a public beta, add a shared Redis/edge rate limiter and set
a spending limit with the selected model provider.

From the separate Android project, build the app against the deployed HTTPS
FastAPI origin:

```shell
./gradlew -PAI_BASE_URL=https://api.example.com assembleDebug
```

## Calendar workflow

Import the General Roman calendar for a civil year as a draft:

```shell
uv run python -m app.cli import-year 2026
```

The importer requests LitCal with `year_type=CIVIL`, drops vigil alternatives
from the default daytime calendar, preserves optional celebrations, stores only
reading references, and checks that all civil dates are present. When coincident
celebrations are all optional, LitCal's first option is the initial display choice;
all options and their optional rank remain visible for user selection.

Drafts are never returned by public endpoints. After a qualified reviewer has
compared every date and the content/redistribution position is recorded, publish
the exact immutable draft:

```shell
uv run python -m app.cli publish RELEASE_ID \
  --reviewer "REVIEWER NAME" \
  --reviewed-at 2026-09-13 \
  --permission-note "DOCUMENTED PERMISSION OR SOURCE-LIMITATION NOTE"
```

Publishing copies a release to `data/published`; it does not edit or delete the
draft, so rollback is simply selecting/removing a published release through the
future operations layer.

Export a published bundle for the Android offline asset:

```shell
uv run python -m app.cli export-android RELEASE_ID \
  OUTPUT_PATH/general-roman-2026.json
```

## Public endpoints

- `GET /health`
- `GET /health/ai`
- `POST /v1/ai/ask`
- `GET /v1/calendar-scopes`
- `GET /v1/liturgy/day?date=2026-09-13&calendar_id=general-roman`
- `GET /v1/content/manifest`
- `GET /v1/content/releases/{release_id}/bundle`

LitCal is a third-party implementation, not a Vatican-operated API. Its output
must be verified before this app describes a year as supported.
