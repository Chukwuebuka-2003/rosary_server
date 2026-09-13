from datetime import date

import pytest

from app.litcal import LitCalImportError, _select_primary


def test_optional_memorial_does_not_replace_weekday() -> None:
    weekday = {
        "event_key": "weekday",
        "grade": 0,
        "grade_lcl": "weekday",
        "type": "mobile",
    }
    memorial = {
        "event_key": "saint",
        "grade": 2,
        "grade_lcl": "optional memorial",
        "type": "fixed",
    }

    assert _select_primary([memorial, weekday], date(2026, 1, 7)) == weekday


def test_optional_mobile_saturday_memorial_does_not_replace_weekday() -> None:
    weekday = {
        "event_key": "OrdWeekday4Saturday",
        "grade": 0,
        "grade_lcl": "weekday",
        "type": "mobile",
    }
    memorial = {
        "event_key": "SatMemBVM1",
        "grade": 2,
        "grade_lcl": "optional memorial",
        "type": "mobile",
    }

    assert _select_primary([weekday, memorial], date(2026, 2, 7)) == weekday


def test_obligatory_celebration_replaces_weekday() -> None:
    weekday = {"event_key": "weekday", "grade": 0, "type": "mobile"}
    memorial = {"event_key": "saint", "grade": 3, "type": "fixed"}

    assert _select_primary([weekday, memorial], date(2026, 1, 21)) == memorial


def test_ambiguous_high_rank_is_rejected() -> None:
    events = [
        {"event_key": "one", "grade": 4, "type": "fixed"},
        {"event_key": "two", "grade": 4, "type": "fixed"},
    ]

    with pytest.raises(LitCalImportError, match="ambiguous primary"):
        _select_primary(events, date(2026, 2, 1))


def test_coincident_optional_memorials_preserve_upstream_default() -> None:
    immaculate_heart = {
        "event_key": "ImmaculateHeart",
        "grade": 2,
        "grade_lcl": "optional memorial",
        "type": "mobile",
    }
    saint_anthony = {
        "event_key": "StAnthonyPadua",
        "grade": 2,
        "grade_lcl": "optional memorial",
        "type": "fixed",
    }

    assert _select_primary(
        [immaculate_heart, saint_anthony], date(2026, 6, 13)
    ) == immaculate_heart
