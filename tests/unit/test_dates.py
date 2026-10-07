from datetime import date

import pytest

from graphrag.ingestion.dates import extract_dates, parse_time_range

TODAY = date(2026, 10, 7)


def test_extracts_english_hindi_and_numeric_dates() -> None:
    text = (
        "On 14 October 2025 two workers were injured. From 1 April 2026 data stays in India. "
        "Q3 2025 bulletin, 14/10/2025, 2025-10-20. ३१ जनवरी २०२६ तक, अक्टूबर 2025 में। Revenue for 2024."
    )
    spans = {s.text: (s.start, s.end) for s in extract_dates(text)}
    assert spans["14 October 2025"] == (date(2025, 10, 14), date(2025, 10, 14))
    assert spans["Q3 2025"] == (date(2025, 7, 1), date(2025, 9, 30))
    assert spans["14/10/2025"] == (date(2025, 10, 14), date(2025, 10, 14))  # day first
    assert spans["31 जनवरी 2026"] == (date(2026, 1, 31), date(2026, 1, 31))  # Devanagari digits converted
    assert spans["अक्टूबर 2025"] == (date(2025, 10, 1), date(2025, 10, 31))
    assert spans["2024"] == (date(2024, 1, 1), date(2024, 12, 31))
    assert "October 2025" not in spans  # not re-read inside "14 October 2025"


def test_amounts_are_not_dates() -> None:
    assert extract_dates("₹1,240 crore, 4,800 tonnes, 120 MLD, 2,100 meters, 31/02/2025") == []


@pytest.mark.parametrize(
    ("question", "start", "end"),
    [
        ("What was revenue in 2025?", date(2025, 1, 1), date(2025, 12, 31)),
        ("incidents in Q3 2025", date(2025, 7, 1), date(2025, 9, 30)),
        ("since April 2026", date(2026, 4, 1), date.max),
        ("after 2025", date(2026, 1, 1), date.max),
        ("before 2026", date.min, date(2025, 12, 31)),
        ("until March 2026", date.min, date(2026, 3, 31)),
        ("between March 2025 and June 2025", date(2025, 3, 1), date(2025, 6, 30)),
        ("last 3 months", date(2026, 7, 7), TODAY),
        ("last week", date(2026, 9, 30), TODAY),
        ("this year", date(2026, 1, 1), TODAY),
        ("पिछले 3 दिन", date(2026, 10, 4), TODAY),
        ("इस साल क्या हुआ?", date(2026, 1, 1), TODAY),
        ("2025 के बाद", date(2026, 1, 1), date.max),
        ("2026 से पहले", date.min, date(2025, 12, 31)),
        ("2025 से", date(2025, 1, 1), date.max),
    ],
)
def test_time_ranges_in_questions(question: str, start: date, end: date) -> None:
    found = parse_time_range(question, TODAY)
    assert found is not None and (found.start, found.end) == (start, end)


def test_no_time_expression() -> None:
    assert parse_time_range("Who leads Ganga Roadways?", TODAY) is None
