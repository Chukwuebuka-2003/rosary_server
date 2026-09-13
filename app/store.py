from __future__ import annotations

from datetime import date
from hashlib import sha256
import json
from pathlib import Path

from app.models import ContentRelease, ReleaseStatus, ReviewRecord


class ReleaseNotFoundError(LookupError):
    pass


class ReleaseStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.drafts = root / "drafts"
        self.published = root / "published"
        self.drafts.mkdir(parents=True, exist_ok=True)
        self.published.mkdir(parents=True, exist_ok=True)

    def save_draft(self, release: ContentRelease) -> Path:
        if release.status != ReleaseStatus.DRAFT:
            raise ValueError("save_draft only accepts draft releases")
        path = self.drafts / f"{release.release_id}.json"
        self._write(path, release)
        return path

    def publish(
        self,
        release_id: str,
        reviewer: str,
        reviewed_at: date,
        permission_note: str,
    ) -> Path:
        if not reviewer.strip() or not permission_note.strip():
            raise ValueError("reviewer and permission note are required")
        draft = self._read(self.drafts / f"{release_id}.json")
        scope = draft.bundle.scope.model_copy(
            update={
                "reviewed_at": reviewed_at,
                "permission_status": permission_note.strip(),
            }
        )
        bundle = draft.bundle.model_copy(update={"scope": scope})
        checksum = sha256(
            json.dumps(
                bundle.model_dump(mode="json", by_alias=True),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        release = draft.model_copy(
            update={
                "status": ReleaseStatus.PUBLISHED,
                "checksum_sha256": checksum,
                "review": ReviewRecord(
                    reviewer=reviewer.strip(),
                    reviewed_at=reviewed_at,
                    permission_note=permission_note.strip(),
                ),
                "bundle": bundle,
            }
        )
        path = self.published / f"{release.release_id}.json"
        self._write(path, release)
        return path

    def published_releases(self) -> list[ContentRelease]:
        return sorted(
            (self._read(path) for path in self.published.glob("*.json")),
            key=lambda release: (release.bundle.scope.coverage_start, release.release_id),
        )

    def get_published(self, release_id: str) -> ContentRelease:
        return self._read(self.published / f"{release_id}.json")

    def release_for_day(self, calendar_id: str, requested_date: date) -> ContentRelease:
        matches = [
            release
            for release in self.published_releases()
            if release.bundle.scope.id == calendar_id
            and release.bundle.scope.coverage_start <= requested_date <= release.bundle.scope.coverage_end
        ]
        if not matches:
            raise ReleaseNotFoundError(
                f"no published {calendar_id} release covers {requested_date.isoformat()}"
            )
        return matches[-1]

    @staticmethod
    def _write(path: Path, release: ContentRelease) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            release.model_dump_json(by_alias=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _read(path: Path) -> ContentRelease:
        if not path.is_file():
            raise ReleaseNotFoundError(f"release not found: {path.stem}")
        return ContentRelease.model_validate(json.loads(path.read_text(encoding="utf-8")))
