from app.intent.prompt import SYSTEM_PROMPT, build_intent_json_schema
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


def test_every_entity_field_has_a_description():
    # Verified live against the real API: without per-field descriptions,
    # the model reliably conflated compare_type (what KIND of thing is
    # being compared) with dimension (the breakdown axis) -- e.g. for
    # "Compare Keytruda vs Opdivo by phase" it returned compare_type="phase"
    # instead of compare_type="drug", dimension="phase".
    entities_schema = build_intent_json_schema()["schema"]["properties"]["entities"]
    for name, prop in entities_schema["properties"].items():
        assert prop.get("description"), f"{name} is missing a description"


def test_compare_type_vs_dimension_are_distinguished_in_their_descriptions():
    entities_schema = build_intent_json_schema()["schema"]["properties"]["entities"]
    compare_type_desc = entities_schema["properties"]["compare_type"]["description"]
    dimension_desc = entities_schema["properties"]["dimension"]["description"]
    assert "breakdown axis" in dimension_desc
    assert "not the breakdown axis" in compare_type_desc.lower()


def test_notes_field_has_a_description_requiring_non_empty():
    schema = build_intent_json_schema()["schema"]
    assert "never empty" in schema["properties"]["notes"]["description"].lower()


def test_system_prompt_forbids_inventing_year_bounds():
    # Verified live: "since 2015" (no stated end) was getting an invented
    # end_year (e.g. 2023) from the model before this rule was added.
    assert "never invent start_year/end_year" in SYSTEM_PROMPT.lower()


def test_system_prompt_requires_non_empty_notes():
    # Verified live: notes came back as an empty string on every query
    # before this rule was added.
    assert "notes must never be empty" in SYSTEM_PROMPT.lower()


def test_system_prompt_clarifies_compare_type_vs_dimension():
    assert "compare_type" in SYSTEM_PROMPT and "dimension" in SYSTEM_PROMPT


def test_analysis_type_has_a_description_distinguishing_geographic_from_count():
    # Verified live: "Which countries have the most recruiting trials for
    # diabetes?" (the assignment appendix's own geographic example) got
    # classified as analysis_type=count before this description existed --
    # the model latched onto "how many" and dropped the country breakdown
    # entirely. The description must explicitly call this out.
    description = build_intent_json_schema()["schema"]["properties"]["analysis_type"]["description"]
    assert "which countries have the most" in description.lower()
    assert "not count" in description.lower()
