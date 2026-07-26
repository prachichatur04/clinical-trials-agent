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
- In particular, never invent start_year/end_year bounds beyond what the query states: "since \
2015" means start_year=2015 and end_year=null (open-ended), NOT an end_year you picked yourself. \
Only set end_year when the query itself gives or clearly implies an upper bound (e.g. "between \
2015 and 2020", "up to 2020").
- Structured fields provided by the caller are ground truth -- do not re-derive them.
- comparison requires both compare_a and compare_b; if only one, downgrade to distribution.
- For a comparison query, compare_type is WHAT KIND of thing is being compared (drug, condition, \
or sponsor) -- not the breakdown axis. "Compare Keytruda vs Opdivo by phase" means \
compare_type="drug" (Keytruda and Opdivo are drugs) and dimension="phase" (phase is how the \
comparison is broken down), never the reverse.
- For network queries, set dimension to "drug_cooccurrence" when the question is about which \
drugs co-occur/combine together (drug<->drug); leave dimension unset for a general sponsor<->drug \
network.
- notes must never be empty: always give a one-sentence plain-English interpretation of what the \
query is asking for -- this is shown directly to the user, not just used internally.
- If the query is ambiguous, pick the most reasonable type and set confidence: low."""

_ENTITY_DESCRIPTIONS = {
    "drug_name": "The drug/intervention name mentioned in the query, if any.",
    "condition": "The medical condition/disease mentioned in the query, if any.",
    "trial_phase": "The trial phase(s) mentioned, e.g. 'PHASE1' or 'PHASE1,PHASE2'.",
    "sponsor": "The trial sponsor name mentioned, if any.",
    "country": "The country/location mentioned, if any.",
    "status": "The trial status mentioned, e.g. 'RECRUITING', 'COMPLETED'.",
    "dimension": (
        "The breakdown axis for a distribution/comparison query -- one of: "
        "phase, status, sponsor_class, sponsor_name, intervention_type, country, "
        "or drug_cooccurrence (for a drug<->drug network). Not what is being compared."
    ),
    "compare_a": "For a comparison query: the first thing being compared (e.g. a drug name).",
    "compare_b": "For a comparison query: the second thing being compared (e.g. a drug name).",
    "compare_type": (
        "For a comparison query: what KIND of entity compare_a/compare_b are -- "
        "one of drug, condition, sponsor. Not the breakdown axis (that's dimension)."
    ),
}
_ENTITY_STRING_FIELDS = list(_ENTITY_DESCRIPTIONS.keys())
_ENTITY_INT_FIELDS = ["start_year", "end_year"]
_ENTITY_INT_DESCRIPTIONS = {
    "start_year": "Start of an explicit or clearly-implied year range. Never invented.",
    "end_year": "End of an explicit year range. Left null for an open-ended range ('since X').",
}


def build_intent_json_schema() -> dict:
    """JSON schema for OpenAI Structured Outputs (strict mode).

    Built from the actual enums in schemas/intent.py so it can't drift from
    the Intent model; the Entities field list is spelled out explicitly
    since strict mode requires every property listed in `required` (nullable
    fields still have to be present, just typed to allow null). Per-field
    `description`s matter more than they might look: without them, the model
    has no signal for what distinguishes e.g. compare_type from dimension,
    and reliably conflates the two.
    """
    entity_properties = {
        name: {"type": ["string", "null"], "description": desc} for name, desc in _ENTITY_DESCRIPTIONS.items()
    }
    entity_properties.update(
        {name: {"type": ["integer", "null"], "description": desc} for name, desc in _ENTITY_INT_DESCRIPTIONS.items()}
    )

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
                "notes": {
                    "type": "string",
                    "description": "A one-sentence plain-English interpretation of the query. Never empty.",
                },
                "confidence": {"type": "string", "enum": [c.value for c in Confidence]},
            },
            "required": ["analysis_type", "entities", "suggested_viz", "query_plan", "notes", "confidence"],
            "additionalProperties": False,
        },
    }
