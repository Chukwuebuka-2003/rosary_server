from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from typing import Any

import httpx

from app.models import (
    CalendarBundle,
    CalendarScope,
    CelebrationOption,
    ContentRelease,
    LiturgicalDay,
    ReadingReference,
    ReleaseStatus,
)


LITCAL_API_BASE = "https://litcal.johnromanodorazio.com/api/v5"
CALENDAR_ID = "general-roman"
READING_FIELDS = (
    ("first_reading", "First reading"),
    ("responsorial_psalm", "Responsorial psalm"),
    ("second_reading", "Second reading"),
    ("gospel_acclamation", "Gospel acclamation"),
    ("gospel", "Gospel"),
    ("dawn", "Mass at dawn"),
    ("day", "Mass during the day"),
    ("night", "Mass during the night"),
)


class LitCalImportError(ValueError):
    pass


class LitCalClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def fetch_general_roman_year(self, year: int) -> dict[str, Any]:
        if not 1970 <= year <= 9999:
            raise ValueError("year must be between 1970 and 9999")
        path = f"{LITCAL_API_BASE}/calendar/{year}"
        params = {"locale": "en", "year_type": "CIVIL"}
        if self._client is not None:
            response = await self._client.get(path, params=params)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(path, params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("litcal"), list):
            raise LitCalImportError("LitCal returned an unexpected response shape")
        return payload


def import_general_roman_year(payload: dict[str, Any], year: int) -> ContentRelease:
    raw_events = payload.get("litcal")
    if not isinstance(raw_events, list):
        raise LitCalImportError("payload must contain a litcal event list")

    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for event in raw_events:
        if not isinstance(event, dict) or event.get("is_vigil_mass") is True:
            continue
        event_date = _parse_event_date(event.get("date"))
        if event_date.year == year:
            by_date[event_date].append(event)

    expected_dates = _dates_for_year(year)
    missing_dates = sorted(set(expected_dates) - set(by_date))
    if missing_dates:
        sample = ", ".join(value.isoformat() for value in missing_dates[:5])
        raise LitCalImportError(f"calendar is missing {len(missing_dates)} civil dates: {sample}")

    upstream_version = str(payload.get("metadata", {}).get("version", "unknown"))
    release_seed = {
        "calendar": CALENDAR_ID,
        "year": year,
        "upstreamVersion": upstream_version,
        "events": raw_events,
    }
    source_checksum = sha256(_canonical_json(release_seed)).hexdigest()
    release_id = f"grc-en-{year}-{source_checksum[:12]}"
    source_url = f"{LITCAL_API_BASE}/calendar/{year}?locale=en&year_type=CIVIL"

    days: list[LiturgicalDay] = []
    celebrations: list[CelebrationOption] = []
    readings: list[ReadingReference] = []
    for event_date in expected_dates:
        events = by_date[event_date]
        primary = _select_primary(events, event_date)
        primary_key = str(primary["event_key"])
        colors = primary.get("color") or []
        color = str(colors[0]) if isinstance(colors, list) and colors else "unknown"
        season = str(primary.get("liturgical_season_lcl") or primary.get("liturgical_season") or "Unknown")
        days.append(
            LiturgicalDay(
                calendar_id=CALENDAR_ID,
                date=event_date,
                season=season,
                liturgical_color=color,
                content_release_id=release_id,
            )
        )

        for event in sorted(events, key=_event_sort_key):
            event_key = str(event.get("event_key") or "unknown")
            celebration_id = f"{CALENDAR_ID}:{event_date.isoformat()}:{event_key}"
            celebrations.append(
                CelebrationOption(
                    id=celebration_id,
                    calendar_id=CALENDAR_ID,
                    date=event_date,
                    title=str(event.get("name") or event_key),
                    rank=str(event.get("grade_lcl") or event.get("grade_display") or "unclassified"),
                    is_primary=event_key == primary_key,
                )
            )
            readings.extend(_map_readings(event, celebration_id, source_url))

    scope = CalendarScope(
        id=CALENDAR_ID,
        display_name="General Roman Calendar",
        rite="Roman Rite (ordinary form)",
        region="Universal calendar only; national and diocesan observances excluded",
        coverage_start=date(year, 1, 1),
        coverage_end=date(year, 12, 31),
        source_name="Liturgical Calendar API (LitCal)",
        source_url=source_url,
        permission_status="Imported for review; publication permission and calendar comparison pending",
        content_version=release_id,
        reviewed_at=None,
    )
    bundle = CalendarBundle(
        scope=scope,
        days=days,
        celebrations=celebrations,
        readings=readings,
    )
    bundle_checksum = sha256(_canonical_json(bundle.model_dump(mode="json", by_alias=True))).hexdigest()
    return ContentRelease(
        release_id=release_id,
        status=ReleaseStatus.DRAFT,
        imported_at=datetime.now(timezone.utc),
        upstream_version=upstream_version,
        checksum_sha256=bundle_checksum,
        bundle=bundle,
    )


def _select_primary(events: list[dict[str, Any]], event_date: date) -> dict[str, Any]:
    if not events:
        raise LitCalImportError(f"no daytime celebration for {event_date.isoformat()}")
    highest_grade = max(_grade(event) for event in events)
    if highest_grade >= 3:
        highest = [event for event in events if _grade(event) == highest_grade]
        if len(highest) != 1:
            keys = [event.get("event_key") for event in highest]
            raise LitCalImportError(
                f"ambiguous primary celebrations for {event_date.isoformat()}: {keys}"
            )
        return highest[0]

    # Optional memorials and commemorations do not replace the temporal weekday
    # automatically. Movement type is not sufficient because the optional
    # Saturday memorial of the Blessed Virgin Mary is also marked as mobile.
    weekdays = [
        event
        for event in events
        if str(event.get("grade_lcl", "")).casefold() == "weekday"
    ]
    if len(weekdays) == 1:
        return weekdays[0]
    if len(events) == 1:
        return events[0]

    # Coincident memorials can both be reduced to optional memorials, leaving no
    # unique obligatory option in LitCal's response. Preserve all options and use
    # upstream order only as the initial display choice. The rank remains visible.
    return events[0]


def _map_readings(
    event: dict[str, Any], celebration_id: str, source_url: str
) -> list[ReadingReference]:
    raw = event.get("readings")
    values: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        values = [
            (label, str(raw.get(key, "")).strip())
            for key, label in READING_FIELDS
            if str(raw.get(key, "")).strip()
        ]
    elif isinstance(raw, str) and raw.strip():
        values = [("Reading selection", raw.strip())]

    return [
        ReadingReference(
            id=f"{celebration_id}:reading:{index}",
            celebration_id=celebration_id,
            order_index=index,
            label=label,
            citation=citation,
            permitted_text=None,
            source_document_title="LitCal reading reference",
            source_url=source_url,
        )
        for index, (label, citation) in enumerate(values)
    ]


def _event_sort_key(event: dict[str, Any]) -> tuple[int, str]:
    return (-_grade(event), str(event.get("event_key", "")))


def _grade(event: dict[str, Any]) -> int:
    try:
        return int(event.get("grade", 0))
    except (TypeError, ValueError):
        return 0


def _parse_event_date(value: Any) -> date:
    if not isinstance(value, str):
        raise LitCalImportError(f"invalid event date: {value!r}")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError as error:
        raise LitCalImportError(f"invalid event date: {value}") from error


def _dates_for_year(year: int) -> list[date]:
    first = date(year, 1, 1).toordinal()
    last = date(year, 12, 31).toordinal()
    return [date.fromordinal(ordinal) for ordinal in range(first, last + 1)]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
