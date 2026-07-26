"""Intent/entities -> CTGov v2 query params.

Only ever emits keys from the allow-list below, built entirely from named
kwargs -- there is no passthrough path for arbitrary caller-supplied param
names, so there is nothing to sanitize at this layer.

Two live-API corrections to the params table in the build plan (verified
2026-07-26 against https://clinicaltrials.gov/api/v2):
  - `filter.phase` does not exist. Phase filtering goes through
    `filter.advanced` with Essie syntax: AREA[Phase](PHASE1 OR PHASE2).
  - `/stats/field/values` rejects any query.*/filter.* param outright
    (400 "Invalid prefix in parameter name"), so it cannot be used for
    scoped counting -- confirming the plan's paginated-fetch default mode
    is the only viable approach for counting.
"""

DEFAULT_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "StartDate",
    "StudyFirstPostDate",
    "Phase",
    "LeadSponsorName",
    "LeadSponsorClass",
    "CollaboratorName",
    "Condition",
    "InterventionType",
    "InterventionName",
    "LocationCountry",
]


def build_query_params(
    *,
    drug_name: str | None = None,
    condition: str | None = None,
    sponsor: str | None = None,
    country: str | None = None,
    status: str | list[str] | None = None,
    phases: str | list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    fields: list[str] | None = None,
    page_size: int = 1000,
    page_token: str | None = None,
) -> dict[str, str]:
    params: dict[str, str] = {"countTotal": "true", "pageSize": str(page_size)}

    if drug_name:
        params["query.intr"] = drug_name
    if condition:
        params["query.cond"] = condition
    if sponsor:
        params["query.spons"] = sponsor
    if country:
        params["query.locn"] = country
    if status:
        params["filter.overallStatus"] = status if isinstance(status, str) else ",".join(status)

    advanced_clauses = _build_advanced_clauses(phases=phases, start_year=start_year, end_year=end_year)
    if advanced_clauses:
        params["filter.advanced"] = " AND ".join(advanced_clauses)

    params["fields"] = "|".join(fields or DEFAULT_FIELDS)

    if page_token:
        params["pageToken"] = page_token

    return params


def _build_advanced_clauses(
    *,
    phases: str | list[str] | None,
    start_year: int | None,
    end_year: int | None,
) -> list[str]:
    clauses = []

    if phases:
        phase_list = [p.strip() for p in phases.split(",")] if isinstance(phases, str) else list(phases)
        clauses.append(f"AREA[Phase]({' OR '.join(phase_list)})")

    if start_year is not None or end_year is not None:
        lower = str(start_year) if start_year is not None else "MIN"
        upper = str(end_year) if end_year is not None else "MAX"
        clauses.append(f"AREA[StartDate]RANGE[{lower},{upper}]")

    return clauses
