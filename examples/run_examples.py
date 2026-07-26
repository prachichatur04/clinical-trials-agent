"""Captures actual JSON outputs for 5 canonical queries, one per major
analysis type, against the live ClinicalTrials.gov API. Run from the repo
root:

    python examples/run_examples.py

Uses whatever OPENAI_API_KEY is configured in .env; falls back to the
heuristic classifier automatically if none is set (still produces valid,
if lower-confidence, output -- see meta.intent_source in each file).
"""

import asyncio
import json
from pathlib import Path

from app.pipeline import run_pipeline
from app.schemas.request import QueryRequest

OUTPUT_DIR = Path(__file__).parent / "outputs"

EXAMPLES: list[tuple[str, QueryRequest]] = [
    (
        "01_trend_pembrolizumab",
        QueryRequest(
            query="How has the number of trials for pembrolizumab changed per year since 2015?",
            drug_name="pembrolizumab",
            start_year=2015,
            max_studies=200,
            include_summary=True,
        ),
    ),
    (
        "02_distribution_lung_cancer",
        QueryRequest(
            query="How are lung cancer trials distributed across phases?",
            condition="lung cancer",
            max_studies=200,
            include_summary=True,
        ),
    ),
    (
        "03_comparison_keytruda_opdivo",
        QueryRequest(
            query="Compare phases for trials involving Keytruda vs Opdivo.",
            compare_a="Keytruda",
            compare_b="Opdivo",
            compare_type="drug",
            dimension="phase",
            max_studies=200,
            include_summary=True,
        ),
    ),
    (
        "04_geographic_diabetes",
        QueryRequest(
            query="Which countries have the most recruiting trials for diabetes?",
            condition="diabetes",
            max_studies=200,
            include_summary=True,
        ),
    ),
    (
        "05_network_breast_cancer",
        QueryRequest(
            query="Show a network of sponsors and drugs for breast cancer trials.",
            condition="breast cancer",
            max_studies=200,
            include_summary=True,
        ),
    ),
]


async def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, request in EXAMPLES:
        print(f"Running {name}: {request.query!r}")
        response = await run_pipeline(request)
        output_path = OUTPUT_DIR / f"{name}.json"
        output_path.write_text(json.dumps(response.model_dump(mode="json"), indent=2) + "\n")
        print(f"  intent_source={response.meta.intent_source.value} matched={response.meta.total_studies_matched}")
        print(f"  -> {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
