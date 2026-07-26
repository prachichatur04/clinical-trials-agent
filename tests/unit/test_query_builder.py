from app.ctgov.query_builder import DEFAULT_FIELDS, build_query_params


def test_always_sets_count_total_and_page_size():
    params = build_query_params()
    assert params["countTotal"] == "true"
    assert params["pageSize"] == "1000"


def test_default_fields_used_when_not_specified():
    params = build_query_params()
    assert params["fields"] == "|".join(DEFAULT_FIELDS)


def test_custom_fields_override_default():
    params = build_query_params(fields=["NCTId", "Phase"])
    assert params["fields"] == "NCTId|Phase"


def test_drug_name_maps_to_query_intr():
    params = build_query_params(drug_name="Pembrolizumab")
    assert params["query.intr"] == "Pembrolizumab"


def test_condition_maps_to_query_cond():
    params = build_query_params(condition="lung cancer")
    assert params["query.cond"] == "lung cancer"


def test_sponsor_maps_to_query_spons():
    params = build_query_params(sponsor="Pfizer")
    assert params["query.spons"] == "Pfizer"


def test_country_maps_to_query_locn():
    params = build_query_params(country="Germany")
    assert params["query.locn"] == "Germany"


def test_status_string_maps_to_filter_overall_status():
    params = build_query_params(status="RECRUITING")
    assert params["filter.overallStatus"] == "RECRUITING"


def test_status_list_joined_with_commas():
    params = build_query_params(status=["RECRUITING", "COMPLETED"])
    assert params["filter.overallStatus"] == "RECRUITING,COMPLETED"


def test_no_advanced_filter_when_no_phase_or_years():
    params = build_query_params(drug_name="x")
    assert "filter.advanced" not in params


def test_single_phase_uses_area_phase_syntax():
    # filter.phase does not exist on the live API -- confirmed via manual probe.
    params = build_query_params(phases="PHASE1")
    assert params["filter.advanced"] == "AREA[Phase](PHASE1)"


def test_phase_list_ored_together():
    params = build_query_params(phases=["PHASE1", "PHASE2"])
    assert params["filter.advanced"] == "AREA[Phase](PHASE1 OR PHASE2)"


def test_comma_separated_phase_string_is_split():
    params = build_query_params(phases="PHASE1,PHASE2")
    assert params["filter.advanced"] == "AREA[Phase](PHASE1 OR PHASE2)"


def test_start_year_only_uses_max_upper_bound():
    params = build_query_params(start_year=2015)
    assert params["filter.advanced"] == "AREA[StartDate]RANGE[2015,MAX]"


def test_end_year_only_uses_min_lower_bound():
    params = build_query_params(end_year=2020)
    assert params["filter.advanced"] == "AREA[StartDate]RANGE[MIN,2020]"


def test_start_and_end_year_both_bound():
    params = build_query_params(start_year=2015, end_year=2020)
    assert params["filter.advanced"] == "AREA[StartDate]RANGE[2015,2020]"


def test_phase_and_year_range_anded_together():
    params = build_query_params(phases="PHASE1", start_year=2015)
    assert params["filter.advanced"] == "AREA[Phase](PHASE1) AND AREA[StartDate]RANGE[2015,MAX]"


def test_page_token_included_when_present():
    params = build_query_params(page_token="abc123")
    assert params["pageToken"] == "abc123"


def test_page_token_omitted_when_absent():
    params = build_query_params()
    assert "pageToken" not in params


def test_only_allow_listed_keys_ever_produced():
    allow_list = {
        "countTotal",
        "pageSize",
        "pageToken",
        "fields",
        "query.intr",
        "query.cond",
        "query.spons",
        "query.locn",
        "filter.overallStatus",
        "filter.advanced",
    }
    params = build_query_params(
        drug_name="x",
        condition="y",
        sponsor="z",
        country="w",
        status="RECRUITING",
        phases="PHASE1",
        start_year=2015,
        end_year=2020,
        page_token="tok",
    )
    assert set(params.keys()) <= allow_list
