"""Date mentions in text (English + Hindi) and time ranges in questions.

Every mention becomes a span [start, end]: "14 October 2025" is one day, "October 2025" a month,
"Q3 2025" a quarter, "2025" a year. Chunks store the overall span of their mentions; questions with a
time expression ("in 2025", "since April 2026", "last 3 months", "पिछले 3 दिन") become a range filter.
Deterministic on purpose: exact, testable, and it works without an LLM.
"""

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

MONTHS: dict[str, int] = {}
for _i, _names in enumerate(
    [
        ("january", "jan", "जनवरी"),
        ("february", "feb", "फ़रवरी", "फरवरी"),
        ("march", "mar", "मार्च"),
        ("april", "apr", "अप्रैल", "अप्रेल"),
        ("may", "मई"),
        ("june", "jun", "जून"),
        ("july", "jul", "जुलाई"),
        ("august", "aug", "अगस्त"),
        ("september", "sep", "sept", "सितंबर", "सितम्बर"),
        ("october", "oct", "अक्टूबर", "अक्तूबर"),
        ("november", "nov", "नवंबर", "नवम्बर"),
        ("december", "dec", "दिसंबर", "दिसम्बर"),
    ],
    start=1,
):
    for _name in _names:
        MONTHS[_name] = _i

_MONTH = "(" + "|".join(sorted(map(re.escape, MONTHS), key=len, reverse=True)) + r")\.?"
_YEAR = r"((?:19|20)\d{2})"
_B = r"(?<![\w.,₹$])"  # not glued to a word, number or currency
_E = r"(?![\w%]|[.,]\d)"

# Most specific first; matched text is blanked out so "14 October 2025" isn't also read as "October 2025"
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ymd", re.compile(_B + r"(\d{4})-(\d{1,2})-(\d{1,2})" + _E)),
    ("dmy_num", re.compile(_B + r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})" + _E)),
    ("d_month_y", re.compile(_B + r"(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH + r",?\s+" + _YEAR + _E, re.I)),
    ("month_d_y", re.compile(_B + _MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+" + _YEAR + _E, re.I)),
    ("quarter", re.compile(_B + r"Q([1-4])\s*(?:FY\s*)?" + _YEAR + _E, re.I)),
    ("month_y", re.compile(_B + _MONTH + r",?\s+" + _YEAR + _E, re.I)),
    ("year", re.compile(_B + _YEAR + _E)),
]


@dataclass(frozen=True)
class DateSpan:
    start: date
    end: date
    text: str


def _month_span(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def _span(kind: str, m: re.Match[str]) -> tuple[date, date] | None:
    g = m.groups()
    try:
        if kind == "ymd":
            d = date(int(g[0]), int(g[1]), int(g[2]))
            return d, d
        if kind == "dmy_num":  # Indian / European order: day first
            d = date(int(g[2]), int(g[1]), int(g[0]))
            return d, d
        if kind == "d_month_y":
            d = date(int(g[2]), MONTHS[g[1].casefold()], int(g[0]))
            return d, d
        if kind == "month_d_y":
            d = date(int(g[2]), MONTHS[g[0].casefold()], int(g[1]))
            return d, d
        if kind == "quarter":
            q, year = int(g[0]), int(g[1])
            return date(year, 3 * q - 2, 1), _month_span(year, 3 * q)[1]
        if kind == "month_y":
            return _month_span(int(g[1]), MONTHS[g[0].casefold()])
        if kind == "year":
            return date(int(g[0]), 1, 1), date(int(g[0]), 12, 31)
    except (ValueError, KeyError):  # 31/02/2025, unknown month...
        return None
    return None


def extract_dates(text: str) -> list[DateSpan]:
    """All date mentions in `text`, in reading order."""
    work = text.translate(_DEVANAGARI_DIGITS)
    found: list[tuple[int, DateSpan]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(work):
            span = _span(kind, m)
            if span:
                found.append((m.start(), DateSpan(span[0], span[1], m.group(0).strip())))
        work = pattern.sub(lambda m: " " * len(m.group(0)), work)  # don't re-read inside a longer match
    return [s for _, s in sorted(found, key=lambda x: x[0])]


def overall_span(spans: list[DateSpan]) -> tuple[date, date] | None:
    if not spans:
        return None
    return min(s.start for s in spans), max(s.end for s in spans)


# ---------- time ranges in questions ----------

@dataclass(frozen=True)
class TimeRange:
    start: date
    end: date
    expression: str  # the words in the question that produced it


_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_HI_UNITS = {
    "दिन": "day", "दिनों": "day", "हफ्ते": "week", "हफ़्ते": "week", "हफ्तों": "week", "सप्ताह": "week",
    "महीने": "month", "महीनों": "month", "साल": "year", "सालों": "year", "वर्ष": "year", "वर्षों": "year",
}
_RELATIVE_EN = re.compile(r"\b(?:last|past|previous)\s+(?:(\d+)\s+)?(day|week|month|year)s?\b", re.I)
_RELATIVE_HI = re.compile(r"(?:पिछले|पिछला|पिछली|बीते)\s+(?:(\d+)\s+)?(" + "|".join(_HI_UNITS) + r")")
_THIS = re.compile(r"\b(?:this|current)\s+(week|month|year)\b|इस\s+(सप्ताह|हफ्ते|महीने|साल|वर्ष)", re.I)
_TODAY = re.compile(r"\btoday\b|आज", re.I)
_YESTERDAY = re.compile(r"\byesterday\b", re.I)  # Hindi "कल" means both yesterday and tomorrow: skipped
_END = r"(?=\s|$|[?.,!।])"  # word end that also works after Devanagari vowel signs (\b does not)


def _relative(n: int, unit: str, today: date) -> tuple[date, date]:
    if unit == "month":
        month = today.month - n
        year = today.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        start = date(year, month, min(today.day, calendar.monthrange(year, month)[1]))
    elif unit == "year":
        start = date(today.year - n, today.month, min(today.day, calendar.monthrange(today.year - n, today.month)[1]))
    else:
        start = today - timedelta(days=n * _UNIT_DAYS[unit])
    return start, today


def parse_time_range(question: str, today: date | None = None) -> TimeRange | None:
    """Time range asked for in a question, or None. Relative expressions use `today` (default: now)."""
    today = today or date.today()
    q = question.translate(_DEVANAGARI_DIGITS)

    for m in (_RELATIVE_EN.search(q), _RELATIVE_HI.search(q)):
        if m:
            unit = m.group(2).lower()
            unit = _HI_UNITS.get(unit, unit)
            start, end = _relative(int(m.group(1) or 1), unit, today)
            return TimeRange(start, end, m.group(0))
    if m := _THIS.search(q):
        unit = (m.group(1) or _HI_UNITS.get(m.group(2), "year")).lower()
        if unit == "week":
            start = today - timedelta(days=today.weekday())
        elif unit == "month":
            start = today.replace(day=1)
        else:
            start = date(today.year, 1, 1)
        return TimeRange(start, today, m.group(0))
    if m := _TODAY.search(q):
        return TimeRange(today, today, m.group(0))
    if m := _YESTERDAY.search(q):
        return TimeRange(today - timedelta(days=1), today - timedelta(days=1), m.group(0))

    spans = extract_dates(question)
    if not spans:
        return None
    first = spans[0]
    pos = q.find(first.text)
    before, after = q[:pos], q[pos + len(first.text):]
    one_day = timedelta(days=1)
    if re.search(r"\bafter\s*$", before, re.I) or re.match(r"\s*के\s+बाद" + _END, after):
        return TimeRange(first.end + one_day, date.max, f"after {first.text}")
    if re.search(r"\b(?:before|prior\s+to)\s*$", before, re.I) or re.match(r"\s*से\s+पहले" + _END, after):
        return TimeRange(date.min, first.start - one_day, f"before {first.text}")
    # checked after "से पहले" (before), which also starts with "से"
    if re.search(r"\b(?:since|from)\s*$", before, re.I) or re.match(r"\s*से" + _END, after):
        return TimeRange(first.start, date.max, f"since {first.text}")
    if re.search(r"\b(?:until|till|up\s+to)\s*$", before, re.I) or re.match(r"\s*तक" + _END, after):
        return TimeRange(date.min, first.end, f"until {first.text}")
    # one or more plain mentions ("in 2025", "between March 2025 and June 2025"): their overall span
    expression = first.text if len(spans) == 1 else f"{first.text} … {spans[-1].text}"
    return TimeRange(min(sp.start for sp in spans), max(sp.end for sp in spans), expression)
