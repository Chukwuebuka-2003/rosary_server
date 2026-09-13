from datetime import date, datetime, timezone
from hashlib import sha256

from fastapi.testclient import TestClient

from app.main import create_app
from app.models import (
    CalendarBundle,
    CalendarScope,
    CelebrationOption,
    ContentRelease,
    LiturgicalDay,
    ReleaseStatus,
)


def release() -> ContentRelease:
    release_id = "grc-en-2026-test"
    bundle = CalendarBundle(
        scope=CalendarScope(
            id="general-roman",
            display_name="General Roman Calendar",
            rite="Roman Rite (ordinary form)",
            region="Universal calendar only; national and diocesan observances excluded",
            coverage_start=date(2026, 1, 1),
            coverage_end=date(2026, 12, 31),
            source_name="Test source",
            source_url="https://example.com/calendar",
            permission_status="Reviewed test fixture",
            content_version=release_id,
            reviewed_at=date(2026, 1, 1),
        ),
        days=[
            LiturgicalDay(
                calendar_id="general-roman",
                date=date(2026, 9, 13),
                season="Ordinary Time",
                liturgical_color="green",
                content_release_id=release_id,
            )
        ],
        celebrations=[
            CelebrationOption(
                id="general-roman:2026-09-13:sunday",
                calendar_id="general-roman",
                date=date(2026, 9, 13),
                title="Twenty-fourth Sunday in Ordinary Time",
                rank="FEAST OF THE LORD",
                is_primary=True,
            )
        ],
        readings=[],
    )
    return ContentRelease(
        release_id=release_id,
        status=ReleaseStatus.PUBLISHED,
        imported_at=datetime.now(timezone.utc),
        upstream_version="test",
        checksum_sha256=sha256(b"test").hexdigest(),
        bundle=bundle,
    )


def test_only_published_releases_are_served(tmp_path) -> None:
    app = create_app(tmp_path)
    app.state.release_store._write(tmp_path / "published" / "grc-en-2026-test.json", release())
    client = TestClient(app)

    scopes = client.get("/v1/calendar-scopes")
    day = client.get(
        "/v1/liturgy/day",
        params={"date": "2026-09-13", "calendar_id": "general-roman"},
    )

    assert scopes.status_code == 200
    assert scopes.json()[0]["displayName"] == "General Roman Calendar"
    assert day.status_code == 200
    assert day.json()["day"]["liturgicalColor"] == "green"


def test_unpublished_date_returns_404(tmp_path) -> None:
    client = TestClient(create_app(tmp_path))

    response = client.get(
        "/v1/liturgy/day",
        params={"date": "2026-09-13", "calendar_id": "general-roman"},
    )

    assert response.status_code == 404


def test_publish_records_review_and_recalculates_bundle_checksum(tmp_path) -> None:
    store = create_app(tmp_path).state.release_store
    draft = release().model_copy(update={"status": ReleaseStatus.DRAFT, "review": None})
    store.save_draft(draft)

    store.publish(
        draft.release_id,
        reviewer="Test Reviewer",
        reviewed_at=date(2026, 9, 13),
        permission_note="Approved test fixture",
    )
    published = store.get_published(draft.release_id)

    assert published.review is not None
    assert published.bundle.scope.reviewed_at == date(2026, 9, 13)
    assert published.checksum_sha256 != draft.checksum_sha256
