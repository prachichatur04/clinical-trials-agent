from app.intent.prompt import build_intent_json_schema
from app.schemas.intent import AnalysisType, Confidence, VizType


def test_schema_is_strict():
    schema = build_intent_json_schema()
    assert schema["strict"] is True
    assert schema["name"] == "intent"


def test_top_level_required_matches_all_top_level_properties():
    schema = build_intent_json_schema()["schema"]
    assert set(schema["required"]) == set(schema["properties"].keys())
    assert schema["additionalProperties"] is False


def test_entities_required_matches_all_entity_properties():
    entities_schema = build_intent_json_schema()["schema"]["properties"]["entities"]
    assert set(entities_schema["required"]) == set(entities_schema["properties"].keys())
    assert entities_schema["additionalProperties"] is False


def test_analysis_type_enum_matches_schemas_intent_enum():
    schema = build_intent_json_schema()["schema"]
    assert set(schema["properties"]["analysis_type"]["enum"]) == {t.value for t in AnalysisType}


def test_suggested_viz_enum_matches_schemas_intent_enum():
    schema = build_intent_json_schema()["schema"]
    assert set(schema["properties"]["suggested_viz"]["enum"]) == {t.value for t in VizType}


def test_confidence_enum_matches_schemas_intent_enum():
    schema = build_intent_json_schema()["schema"]
    assert set(schema["properties"]["confidence"]["enum"]) == {c.value for c in Confidence}


def test_entity_fields_are_nullable_not_merely_optional():
    # Strict mode requires every field in `required` -- nullable fields must
    # still be listed there, just typed to allow null instead of omitted.
    entities_schema = build_intent_json_schema()["schema"]["properties"]["entities"]
    for name, prop in entities_schema["properties"].items():
        assert "null" in prop["type"], f"{name} must allow null under strict mode"
