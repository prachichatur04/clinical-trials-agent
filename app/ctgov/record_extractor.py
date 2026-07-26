from app.schemas.internal import TrialRecord
from app.utils.safe_get import safe_get


def extract_record(study: dict) -> TrialRecord:
    """Flatten one raw CTGov study JSON object into a TrialRecord.

    Every read goes through safe_get -- CTGov omits whole modules (e.g. an
    empty `designModule: {}`) whenever a study simply doesn't have that data,
    not just individual fields.
    """
    nct_id = safe_get(study, "protocolSection.identificationModule.nctId")
    if not nct_id:
        raise ValueError("study is missing protocolSection.identificationModule.nctId")

    locations = safe_get(study, "protocolSection.contactsLocationsModule.locations", default=[]) or []
    countries = _dedupe_preserve_order(
        loc.get("country") for loc in locations if isinstance(loc, dict) and loc.get("country")
    )

    interventions = safe_get(study, "protocolSection.armsInterventionsModule.interventions", default=[]) or []
    intervention_types = [i["type"] for i in interventions if isinstance(i, dict) and i.get("type")]
    intervention_names = [i["name"] for i in interventions if isinstance(i, dict) and i.get("name")]

    collaborators = safe_get(study, "protocolSection.sponsorCollaboratorsModule.collaborators", default=[]) or []
    collaborator_names = [c["name"] for c in collaborators if isinstance(c, dict) and c.get("name")]

    return TrialRecord(
        nct_id=nct_id,
        brief_title=safe_get(study, "protocolSection.identificationModule.briefTitle"),
        overall_status=safe_get(study, "protocolSection.statusModule.overallStatus"),
        start_date=safe_get(study, "protocolSection.statusModule.startDateStruct.date"),
        study_first_post_date=safe_get(study, "protocolSection.statusModule.studyFirstPostDateStruct.date"),
        phases=safe_get(study, "protocolSection.designModule.phases", default=[]) or [],
        lead_sponsor_name=safe_get(study, "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"),
        lead_sponsor_class=safe_get(study, "protocolSection.sponsorCollaboratorsModule.leadSponsor.class"),
        collaborator_names=collaborator_names,
        conditions=safe_get(study, "protocolSection.conditionsModule.conditions", default=[]) or [],
        intervention_types=intervention_types,
        intervention_names=intervention_names,
        countries=countries,
        raw=study,
    )


def _dedupe_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
