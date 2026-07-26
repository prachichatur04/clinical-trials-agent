import json
from pathlib import Path

import pytest

from app.ctgov.record_extractor import extract_record

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample_studies.json"


@pytest.fixture
def real_studies():
    return json.loads(FIXTURE_PATH.read_text())["studies"]


def test_extracts_nct_id_and_title(real_studies):
    record = extract_record(real_studies[0])
    assert record.nct_id == "NCT02113852"
    assert "Lung Cancer" in record.brief_title


def test_empty_design_module_yields_empty_phases(real_studies):
    # real_studies[0] has "designModule": {} in the fixture.
    record = extract_record(real_studies[0])
    assert record.phases == []


def test_missing_locations_module_yields_empty_countries(real_studies):
    record = extract_record(real_studies[0])
    assert record.countries == []


def test_missing_collaborators_yields_empty_list(real_studies):
    record = extract_record(real_studies[0])
    assert record.collaborator_names == []


def test_present_collaborators_extracted(real_studies):
    # real_studies[1] (NCT00165438) has one collaborator in the fixture.
    record = extract_record(real_studies[1])
    assert record.collaborator_names == ["Brigham and Women's Hospital"]


def test_present_phase_extracted(real_studies):
    # real_studies[3] (NCT00985855) has phases: ["PHASE2"] in the fixture.
    record = extract_record(real_studies[3])
    assert record.phases == ["PHASE2"]


def test_study_url_built_from_nct_id(real_studies):
    record = extract_record(real_studies[0])
    assert record.study_url == "https://clinicaltrials.gov/study/NCT02113852"


def test_raw_json_retained_for_citations(real_studies):
    record = extract_record(real_studies[0])
    assert record.raw["protocolSection"]["identificationModule"]["nctId"] == "NCT02113852"


def test_raw_excluded_from_serialization(real_studies):
    record = extract_record(real_studies[0])
    assert "raw" not in record.model_dump()


def test_missing_nct_id_raises():
    with pytest.raises(ValueError):
        extract_record({"protocolSection": {}})


def test_multi_phase_study_extracted_as_list():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999999"},
            "designModule": {"phases": ["PHASE1", "PHASE2"]},
        }
    }
    record = extract_record(study)
    assert record.phases == ["PHASE1", "PHASE2"]


def test_countries_deduped_per_study_across_multiple_sites():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999998"},
            "contactsLocationsModule": {
                "locations": [
                    {"city": "Boston", "country": "United States"},
                    {"city": "Chicago", "country": "United States"},
                    {"city": "Berlin", "country": "Germany"},
                ]
            },
        }
    }
    record = extract_record(study)
    assert record.countries == ["United States", "Germany"]


def test_multiple_conditions_all_kept_not_deduped_to_one():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999997"},
            "conditionsModule": {"conditions": ["Lung Cancer", "Metastatic Disease"]},
        }
    }
    record = extract_record(study)
    assert record.conditions == ["Lung Cancer", "Metastatic Disease"]


def test_intervention_types_and_names_paired_lists():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99999996"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Pembrolizumab"},
                    {"type": "BIOLOGICAL", "name": "Vaccine X"},
                ]
            },
        }
    }
    record = extract_record(study)
    assert record.intervention_types == ["DRUG", "BIOLOGICAL"]
    assert record.intervention_names == ["Pembrolizumab", "Vaccine X"]


def test_entirely_missing_protocol_section_still_raises_cleanly():
    with pytest.raises(ValueError):
        extract_record({})
