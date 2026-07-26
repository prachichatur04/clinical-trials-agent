from collections.abc import Iterator

from app.schemas.internal import TrialRecord
from app.utils.date_parser import extract_year

# (bucket key, nct_id, field path the value came from, verbatim excerpt)
ExtractedValue = tuple[str, str, str, str]

_PHASE_LABELS = {
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "EARLY_PHASE1": "Early Phase 1",
    "NA": "N/A",
}


def by_year(record: TrialRecord) -> Iterator[ExtractedValue]:
    """Trend dimension. Falls back to studyFirstPostDate when startDate is
    missing, per the locked date-handling decision; never drops a record --
    an unparseable/absent date still lands in extract_year's "unknown"
    bucket.
    """
    date_str = record.start_date
    field_path = "protocolSection.statusModule.startDateStruct.date"
    if not date_str:
        date_str = record.study_first_post_date
        field_path = "protocolSection.statusModule.studyFirstPostDateStruct.date"

    year = extract_year(date_str)
    excerpt = date_str if date_str else "no start date recorded"
    yield (year, record.nct_id, field_path, excerpt)


def by_phase(record: TrialRecord) -> Iterator[ExtractedValue]:
    """Multi-phase studies (e.g. ["PHASE1", "PHASE2"]) get their own
    combined category "Phase 1/Phase 2" rather than being counted once per
    phase -- one bucket per record, per the locked multi-phase decision.
    """
    field_path = "protocolSection.designModule.phases"
    if not record.phases:
        yield ("N/A", record.nct_id, field_path, "no phase recorded")
        return

    key = "/".join(_PHASE_LABELS.get(p, p) for p in record.phases)
    excerpt = ", ".join(record.phases)
    yield (key, record.nct_id, field_path, excerpt)


def by_status(record: TrialRecord) -> Iterator[ExtractedValue]:
    field_path = "protocolSection.statusModule.overallStatus"
    if not record.overall_status:
        yield ("UNKNOWN", record.nct_id, field_path, "no status recorded")
        return
    yield (record.overall_status, record.nct_id, field_path, record.overall_status)


def by_sponsor_class(record: TrialRecord) -> Iterator[ExtractedValue]:
    field_path = "protocolSection.sponsorCollaboratorsModule.leadSponsor.class"
    if not record.lead_sponsor_class:
        yield ("UNKNOWN", record.nct_id, field_path, "no sponsor class recorded")
        return
    yield (record.lead_sponsor_class, record.nct_id, field_path, record.lead_sponsor_class)


def by_sponsor_name(record: TrialRecord) -> Iterator[ExtractedValue]:
    field_path = "protocolSection.sponsorCollaboratorsModule.leadSponsor.name"
    if not record.lead_sponsor_name:
        yield ("Unknown Sponsor", record.nct_id, field_path, "no sponsor name recorded")
        return
    yield (record.lead_sponsor_name, record.nct_id, field_path, record.lead_sponsor_name)


def by_country(record: TrialRecord) -> Iterator[ExtractedValue]:
    """One tuple per country (already deduped per study by record_extractor,
    so a study with 5 US sites still yields "United States" once here)."""
    field_path = "protocolSection.contactsLocationsModule.locations"
    if not record.countries:
        yield ("Unknown", record.nct_id, field_path, "no location recorded")
        return
    for country in record.countries:
        yield (country, record.nct_id, field_path, country)


def by_intervention_type(record: TrialRecord) -> Iterator[ExtractedValue]:
    """Multi-count per value: a study with both a DRUG and a BIOLOGICAL
    intervention contributes to both buckets, per the locked multi-value
    decision (unlike by_phase, which combines into one category)."""
    field_path = "protocolSection.armsInterventionsModule.interventions"
    if not record.intervention_types:
        yield ("UNKNOWN", record.nct_id, field_path, "no intervention type recorded")
        return
    for intervention_type in record.intervention_types:
        yield (intervention_type, record.nct_id, field_path, intervention_type)
