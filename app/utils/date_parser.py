import re
from datetime import UTC, date, datetime

from dateutil import parser as dateutil_parser

# CTGov dates observed in practice: "2015-03-14", "2013-08", "2020", or absent.
# Bounds guard against dateutil silently defaulting an ambiguous fragment
# (e.g. a bare "12") to an implausible year.
_MIN_YEAR = 1990
_MAX_YEAR = 2100
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def parse_date(date_str: str | None) -> date | None:
    """Parse a CTGov date string, or return None if it can't be parsed."""
    if not date_str:
        return None
    try:
        parsed = dateutil_parser.parse(date_str, default=datetime(1900, 1, 1, tzinfo=UTC))
    except (ValueError, OverflowError, TypeError):
        return None
    if not (_MIN_YEAR <= parsed.year <= _MAX_YEAR):
        return None
    return parsed.date()


def extract_year(date_str: str | None) -> str:
    """Best-effort year bucket for trend analysis.

    Cascades: full date parse -> regex year extraction -> "unknown". Never
    raises and never drops a record silently, per the plan's date-handling
    decision.
    """
    if not date_str:
        return "unknown"

    parsed = parse_date(date_str)
    if parsed is not None:
        return str(parsed.year)

    match = _YEAR_RE.search(date_str)
    if match:
        return match.group(0)

    return "unknown"
