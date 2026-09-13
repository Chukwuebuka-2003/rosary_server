from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class ReleaseStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"


class CalendarScope(ApiModel):
    id: str
    display_name: str = Field(alias="displayName")
    rite: str
    region: str
    coverage_start: date = Field(alias="coverageStart")
    coverage_end: date = Field(alias="coverageEnd")
    source_name: str = Field(alias="sourceName")
    source_url: HttpUrl = Field(alias="sourceUrl")
    permission_status: str = Field(alias="permissionStatus")
    content_version: str = Field(alias="contentVersion")
    reviewed_at: date | None = Field(default=None, alias="reviewedAt")


class LiturgicalDay(ApiModel):
    calendar_id: str = Field(alias="calendarId")
    date: date
    season: str
    liturgical_color: str = Field(alias="liturgicalColor")
    content_release_id: str = Field(alias="contentReleaseId")


class CelebrationOption(ApiModel):
    id: str
    calendar_id: str = Field(alias="calendarId")
    date: date
    title: str
    rank: str
    is_primary: bool = Field(alias="isPrimary")


class ReadingReference(ApiModel):
    id: str
    celebration_id: str = Field(alias="celebrationId")
    order_index: int = Field(alias="orderIndex", ge=0)
    label: str
    citation: str
    permitted_text: str | None = Field(default=None, alias="permittedText")
    source_document_title: str = Field(alias="sourceDocumentTitle")
    source_url: HttpUrl = Field(alias="sourceUrl")


class CalendarBundle(ApiModel):
    scope: CalendarScope
    days: list[LiturgicalDay]
    celebrations: list[CelebrationOption]
    readings: list[ReadingReference]

    @model_validator(mode="after")
    def validate_relationships(self) -> CalendarBundle:
        day_keys = {(day.calendar_id, day.date) for day in self.days}
        if len(day_keys) != len(self.days):
            raise ValueError("liturgical day keys must be unique")

        primary_count: dict[tuple[str, date], int] = {key: 0 for key in day_keys}
        celebration_ids: set[str] = set()
        for celebration in self.celebrations:
            key = (celebration.calendar_id, celebration.date)
            if key not in day_keys:
                raise ValueError(f"celebration {celebration.id} has no matching day")
            if celebration.id in celebration_ids:
                raise ValueError(f"duplicate celebration id: {celebration.id}")
            celebration_ids.add(celebration.id)
            primary_count[key] += int(celebration.is_primary)

        invalid_days = [key for key, count in primary_count.items() if count != 1]
        if invalid_days:
            raise ValueError(f"days must have exactly one primary celebration: {invalid_days[:5]}")

        reading_positions: set[tuple[str, int]] = set()
        for reading in self.readings:
            if reading.celebration_id not in celebration_ids:
                raise ValueError(f"reading {reading.id} has no matching celebration")
            position = (reading.celebration_id, reading.order_index)
            if position in reading_positions:
                raise ValueError(f"duplicate reading position: {position}")
            reading_positions.add(position)
        return self


class ReviewRecord(ApiModel):
    reviewer: str
    reviewed_at: date = Field(alias="reviewedAt")
    permission_note: str = Field(alias="permissionNote")


class ContentRelease(ApiModel):
    release_id: str = Field(alias="releaseId")
    status: ReleaseStatus
    imported_at: datetime = Field(alias="importedAt")
    upstream_version: str = Field(alias="upstreamVersion")
    checksum_sha256: str = Field(alias="checksumSha256")
    bundle: CalendarBundle
    review: ReviewRecord | None = None


class ReleaseManifestItem(ApiModel):
    release_id: str = Field(alias="releaseId")
    calendar_id: str = Field(alias="calendarId")
    coverage_start: date = Field(alias="coverageStart")
    coverage_end: date = Field(alias="coverageEnd")
    checksum_sha256: str = Field(alias="checksumSha256")
    download_url: str = Field(alias="downloadUrl")


class LiturgicalDayResponse(ApiModel):
    scope: CalendarScope
    day: LiturgicalDay
    celebrations: list[CelebrationOption]
    readings: list[ReadingReference]
