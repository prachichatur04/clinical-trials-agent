from app.utils.safe_get import safe_get


def test_returns_value_at_nested_path():
    data = {"a": {"b": {"c": 42}}}
    assert safe_get(data, "a.b.c") == 42


def test_returns_default_when_key_missing():
    data = {"a": {"b": {}}}
    assert safe_get(data, "a.b.c", default="none") == "none"


def test_returns_default_when_intermediate_key_missing():
    data = {"a": {}}
    assert safe_get(data, "a.b.c", default="none") == "none"


def test_returns_default_when_root_key_missing():
    data = {}
    assert safe_get(data, "a.b.c", default="none") == "none"


def test_default_is_none_when_not_specified():
    assert safe_get({}, "a.b") is None


def test_raises_typeerror_short_circuits_to_default_when_intermediate_is_not_a_dict():
    data = {"a": "not_a_dict"}
    assert safe_get(data, "a.b.c", default="fallback") == "fallback"


def test_raises_when_intermediate_is_a_list():
    data = {"a": [1, 2, 3]}
    assert safe_get(data, "a.b", default="fallback") == "fallback"


def test_single_segment_path():
    data = {"a": 1}
    assert safe_get(data, "a") == 1


def test_empty_path_returns_default():
    data = {"a": 1}
    assert safe_get(data, "", default="fallback") == "fallback"


def test_none_value_at_path_is_returned_as_is_not_default():
    data = {"a": {"b": None}}
    assert safe_get(data, "a.b", default="fallback") is None
