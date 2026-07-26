from app.utils.date_parser import extract_year, parse_date


def test_parse_date_full_iso():
    assert parse_date("2015-03-14").year == 2015


def test_parse_date_year_month_only():
    d = parse_date("2013-08")
    assert d.year == 2013
    assert d.month == 8


def test_parse_date_year_only():
    assert parse_date("2020").year == 2020


def test_parse_date_none_input_returns_none():
    assert parse_date(None) is None


def test_parse_date_empty_string_returns_none():
    assert parse_date("") is None


def test_parse_date_garbage_returns_none():
    assert parse_date("not a date") is None


def test_extract_year_from_full_date():
    assert extract_year("2015-03-14") == "2015"


def test_extract_year_from_year_month():
    assert extract_year("2013-08") == "2013"


def test_extract_year_from_bare_year():
    assert extract_year("2020") == "2020"


def test_extract_year_none_is_unknown_bucket():
    assert extract_year(None) == "unknown"


def test_extract_year_empty_string_is_unknown_bucket():
    assert extract_year("") == "unknown"


def test_extract_year_falls_back_to_regex_when_unparseable_but_year_embedded():
    assert extract_year("circa 2018, exact date unknown") == "2018"


def test_extract_year_completely_unparseable_is_unknown_never_dropped():
    assert extract_year("N/A") == "unknown"


def test_extract_year_never_raises_on_implausible_bare_number():
    # A bare "12" must not be silently mis-parsed as year 1900 by dateutil's
    # month-only default-filling behavior.
    assert extract_year("12") == "unknown"
