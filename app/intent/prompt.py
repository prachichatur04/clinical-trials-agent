from app.schemas.intent import AnalysisType, Confidence, VizType

SYSTEM_PROMPT = """You are the planning component of a clinical trials analysis agent. Given a \
question, you produce a structured query plan. You do NOT answer the question or compute any \
numbers.

Your job:
1. Classify the analysis type (trend, distribution, comparison, geographic, network, count)
2. Extract entities and filters from the query
3. Write a brief query_plan explaining what data you will retrieve and how you will visualize it
4. Suggest a visualization type

Rules:
- Extract only what is explicit or clearly implied. Never invent entities.
- Structured fields provided by the caller are ground truth -- do not re-derive them.
- comparison requires both compare_a and compare_b; if only one, downgrade to distribution.
- If the query is ambiguous, pick the most reasonable type and set confidence: low."""

_ENTITY_STRING_FIELDS = [
    "drug_name",
    "condition",
    "trial_phase",
    "sponsor",
    "country",
    "status",
    "dimension",
    "compare_a",
    "compare_b",
    "compare_type",
]
_ENTITY_INT_FIELDS = ["start_year", "end_year"]


def build_intent_json_schema() -> dict:
    """JSON schema for OpenAI Structured Outputs (strict mode).

    Built from the actual enums in schemas/intent.py so it can't drift from
    the Intent model; the Entities field list is spelled out explicitly
    since strict mode requires every property listed in `required` (nullable
    fields still have to be present, just typed to allow null).
    """
    entity_properties = {name: {"type": ["string", "null"]} for name in _ENTITY_STRING_FIELDS}
    entity_properties.update({name: {"type": ["integer", "null"]} for name in _ENTITY_INT_FIELDS})

    return {
        "name": "intent",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis_type": {"type": "string", "enum": [t.value for t in AnalysisType]},
                "entities": {
                    "type": "object",
                    "properties": entity_properties,
                    "required": _ENTITY_STRING_FIELDS + _ENTITY_INT_FIELDS,
                    "additionalProperties": False,
                },
                "suggested_viz": {"type": "string", "enum": [t.value for t in VizType]},
                "query_plan": {"type": "string"},
                "notes": {"type": "string"},
                "confidence": {"type": "string", "enum": [c.value for c in Confidence]},
            },
            "required": ["analysis_type", "entities", "suggested_viz", "query_plan", "notes", "confidence"],
            "additionalProperties": False,
        },
    }
