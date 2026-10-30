"""Climate-layer contracts: synthetic parsing plus the independent event table."""
import csv
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from src.climate_data import (
    audit_observation_duplicates,
    audit_observation_rows,
    build_sha256_manifest,
    compare_common_report_types,
    compare_seasons,
    classify_observation_quality,
    haversine_km,
    interpret_isd_quality_flag,
    observation_rows_for_event,
    parse_isd_measurement,
    parse_utc,
    quality_summary_by_common_report_types,
    quality_summary_by_event,
    records_in_window,
    validate_events,
    window_for_event,
    write_json_new,
)

ROOT = Path(__file__).resolve().parents[1]
CLIMATE_DATA = ROOT / "data" / "climate_phase3b_20261022"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_event_table_has_ten_unique_events_and_valid_local_clock_conversions():
    events = read_csv(CLIMATE_DATA / "events.csv")
    assert len(events) == 10
    assert len({row["race_id"] for row in events}) == 10
    assert {row["season"] for row in events} == {"2013", "2014"}
    assert not validate_events(events)
    brazil = [row for row in events if row["grand_prix"] == "Brazilian Grand Prix"]
    assert {row["utc_offset"] for row in brazil} == {"-02:00"}
    assert all(row["actual_start_utc"] == "" for row in events)
    assert all(row["documented_start_local"] for row in events
               if row["grand_prix"] == "Brazilian Grand Prix" and row["season"] == "2014"
               or row["grand_prix"] == "Bahrain Grand Prix" and row["season"] == "2014")


def test_noaa_temperature_and_dewpoint_scale_flags_and_missing_values():
    assert parse_isd_measurement("+0290,1") == {
        "raw": "+0290,1", "value_c": 29.0, "quality_flag": "1", "missing_reason": ""
    }
    assert parse_isd_measurement("-0123,2")["value_c"] == -12.3
    missing = parse_isd_measurement("+9999,9")
    assert missing["value_c"] is None
    assert missing["quality_flag"] == "9"
    assert missing["missing_reason"] == "isd_sentinel_9999"
    assert parse_isd_measurement(None)["missing_reason"] == "field_absent"
    assert parse_isd_measurement("bad,1")["missing_reason"] == "malformed_value_token"


def test_noaa_temperature_quality_flags_match_official_page_11_policy():
    for code in ("1", "5"):
        result = interpret_isd_quality_flag(code, "TMP")
        assert result["accepted_all_qc"] is True
        assert result["quality_class"].startswith("passed_all_qc")
    assert interpret_isd_quality_flag("2", "DEW")["quality_class"] == "suspect"
    assert interpret_isd_quality_flag("3", "TMP")["quality_class"] == "erroneous"
    documented_nonpassing = {
        "0": "gross_limits_only", "4": "gross_limits_only_ncei",
        "6": "suspect_ncei", "7": "erroneous_ncei",
        "9": "gross_limits_if_present", "A": "suspect_accepted_as_good",
        "C": "awos_whole_degree", "I": "inserted_by_validator",
        "M": "manually_changed", "P": "replaced_by_validator",
        "R": "replaced_by_ncei_software", "U": "replaced_with_edited_value",
    }
    for code, classification in documented_nonpassing.items():
        result = interpret_isd_quality_flag(code, "DEW")
        assert result["quality_class"] == classification
        assert result["accepted_all_qc"] is False
    assert interpret_isd_quality_flag("X", "TMP")["quality_class"] == "unknown_code"
    assert interpret_isd_quality_flag(None, "DEW")["quality_class"] == "missing_flag"
    with pytest.raises(ValueError, match="limited to TMP and DEW"):
        interpret_isd_quality_flag("1", "SLP")


def test_quality_filter_is_independent_for_tmp_and_dew_and_preserves_source():
    source = [
        {"race_id": "a", "tmp_raw": "+0100,1", "tmp_quality_flag": "1",
         "air_temperature_c": 10, "dew_raw": "+0050,2", "dew_quality_flag": "2",
         "dew_point_c": 5},
        {"race_id": "a", "tmp_raw": "+0200,2", "tmp_quality_flag": "2",
         "air_temperature_c": 20, "dew_raw": "+0150,5", "dew_quality_flag": "5",
         "dew_point_c": 15},
        {"race_id": "a", "tmp_raw": "+9999,1", "tmp_quality_flag": "1",
         "air_temperature_c": None, "dew_raw": "", "dew_quality_flag": "",
         "dew_point_c": None},
    ]
    before = deepcopy(source)
    classified = classify_observation_quality(source)
    assert source == before
    assert classified[0]["tmp_qc_eligible"] is True
    assert classified[0]["dew_qc_eligible"] is False
    assert classified[1]["tmp_qc_eligible"] is False
    assert classified[1]["dew_qc_eligible"] is True
    assert classified[2]["tmp_qc_eligible"] is False
    assert classified[2]["dew_qc_eligible"] is False
    assert classified[0]["tmp_raw"] == "+0100,1"
    assert classified[0]["dew_raw"] == "+0050,2"


def test_quality_summaries_keep_all_values_and_qc_accepted_values_separate():
    events = [
        {"season": "2013", "grand_prix": "GP", "race_id": "a"},
        {"season": "2014", "grand_prix": "GP", "race_id": "b"},
    ]
    source = [
        {"season": "2013", "grand_prix": "GP", "race_id": "a", "report_type": "FM-15",
         "observation_time_utc": "2013-01-01T08:00:00Z", "distance_to_circuit_km": 5,
         "air_temperature_c": 10, "tmp_quality_flag": "1", "dew_point_c": 5, "dew_quality_flag": "2"},
        {"season": "2013", "grand_prix": "GP", "race_id": "a", "report_type": "SY-MT",
         "observation_time_utc": "2013-01-01T09:00:00Z", "distance_to_circuit_km": 5,
         "air_temperature_c": 20, "tmp_quality_flag": "2", "dew_point_c": 15, "dew_quality_flag": "1"},
        {"season": "2014", "grand_prix": "GP", "race_id": "b", "report_type": "FM-15",
         "observation_time_utc": "2014-01-01T10:00:00Z", "distance_to_circuit_km": 6,
         "air_temperature_c": 30, "tmp_quality_flag": "5", "dew_point_c": 25, "dew_quality_flag": "5"},
    ]
    classified = classify_observation_quality(source)
    summaries = quality_summary_by_event(events, classified)
    first, second = summaries
    assert first["tmp_numeric_count_all"] == 2
    assert first["tmp_qc_accepted_count"] == 1
    assert first["tmp_mean_all_c"] == 15
    assert first["tmp_mean_qc_accepted_c"] == 10
    assert first["dew_qc_accepted_count"] == 1
    assert first["dew_mean_all_c"] == 10
    assert first["dew_mean_qc_accepted_c"] == 15
    assert second["tmp_mean_qc_accepted_c"] == 30
    common = quality_summary_by_common_report_types(events, classified)
    assert len(common) == 1
    assert common[0]["report_type"] == "FM-15"
    assert common[0]["2013_tmp_qc_accepted_count"] == 1
    assert common[0]["2014_tmp_qc_accepted_count"] == 1
    assert common[0]["comparison_scope"].find("timestamps not matched") >= 0


def test_window_is_fixed_half_open_utc_and_input_rows_are_not_mutated():
    start, end = window_for_event("2014-03-30T08:00:00Z")
    assert start.isoformat() == "2014-03-30T06:00:00+00:00"
    assert end.isoformat() == "2014-03-30T12:00:00+00:00"
    source = [
        {"DATE": "2014-03-30T12:00:00", "REPORT_TYPE": "FM-15"},
        {"DATE": "2014-03-30T06:00:00", "REPORT_TYPE": "FM-15"},
        {"DATE": "2014-03-30T08:00:00", "REPORT_TYPE": "FM-15"},
        {"DATE": "2014-03-30T05:59:59", "REPORT_TYPE": "FM-15"},
    ]
    before = deepcopy(source)
    selected = records_in_window(source, "2014-03-30T08:00:00Z")
    assert [row["DATE"] for row in selected] == ["2014-03-30T06:00:00", "2014-03-30T08:00:00"]
    assert source == before


def test_duplicate_timestamp_reports_are_audited_without_dropping_rows():
    records = [
        {"STATION": "x", "DATE": "2014-03-30T08:00:00", "REPORT_TYPE": "FM-15"},
        {"STATION": "x", "DATE": "2014-03-30T08:00:00", "REPORT_TYPE": "FM-15"},
        {"STATION": "x", "DATE": "2014-03-30T08:00:00", "REPORT_TYPE": "SY-MT"},
    ]
    audit = audit_observation_duplicates(records)
    assert audit["duplicate_key_count"] == 1
    assert audit["multiple_report_type_timestamps"] == [
        ("2014-03-30T08:00:00", ["FM-15", "SY-MT"])
    ]
    assert len(records) == 3


def test_station_identity_and_distance_are_validated():
    event = {
        "season": "2014", "grand_prix": "Test GP", "race_id": "1", "circuit_id": "1",
        "scheduled_start_utc": "2014-01-01T08:00:00Z", "circuit_latitude": "0",
        "circuit_longitude": "0",
    }
    station = {"station_id": "12345678901", "station_name": "Test",
               "station_latitude": 0, "station_longitude": 1, "station_elevation_m": 10}
    records = [{"STATION": "99999999999", "DATE": "2014-01-01T08:00:00",
                "REPORT_TYPE": "FM-15", "TMP": "+0100,1", "DEW": "+0050,1"}]
    with pytest.raises(ValueError, match="station ID mismatch"):
        observation_rows_for_event(event, station, records, "https://example.invalid", "2026-10-22T00:00:00Z", "hash")
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=0.01)
    assert haversine_km(0, 0, 0, 0) == 0
    with pytest.raises(ValueError, match="latitude"):
        haversine_km(91, 0, 0, 0)


def test_observations_preserve_quality_flags_and_compute_station_distance():
    event = {
        "season": "2014", "grand_prix": "Test GP", "race_id": "1", "circuit_id": "1",
        "scheduled_start_utc": "2014-01-01T08:00:00Z", "circuit_latitude": "0",
        "circuit_longitude": "0",
    }
    station = {"station_id": "12345678901", "station_name": "Test",
               "station_latitude": 0, "station_longitude": 1, "station_elevation_m": 10}
    records = [{"STATION": "12345678901", "DATE": "2014-01-01T08:00:00",
                "REPORT_TYPE": "FM-15", "TMP": "+0100,1", "DEW": "+0050,2"}]
    rows = observation_rows_for_event(
        event, station, records, "https://example.invalid/query", "2026-10-22T00:00:00Z", "abc"
    )
    assert len(rows) == 1
    assert rows[0]["air_temperature_c"] == 10
    assert rows[0]["dew_point_c"] == 5
    assert rows[0]["tmp_quality_flag"] == "1"
    assert rows[0]["dew_quality_flag"] == "2"
    assert rows[0]["quality_flags_interpreted"] is False
    assert rows[0]["distance_to_circuit_km"] == pytest.approx(111.195, abs=0.01)


def test_comparison_is_descriptive_and_keeps_missing_season_unfilled():
    events = [
        {"season": "2013", "grand_prix": "GP", "race_id": "a",
         "scheduled_start_utc": "2013-01-01T08:00:00Z",
         "scheduled_start_local": "2013-01-01T10:00:00+02:00", "utc_offset": "+02:00"},
        {"season": "2014", "grand_prix": "GP", "race_id": "b",
         "scheduled_start_utc": "2014-01-01T08:00:00Z",
         "scheduled_start_local": "2014-01-01T10:00:00+02:00", "utc_offset": "+02:00"},
    ]
    observations = [
        {"race_id": "a", "observation_time_utc": "2013-01-01T07:00:00Z",
         "air_temperature_c": 20, "dew_point_c": 10, "distance_to_circuit_km": 30},
        {"race_id": "a", "observation_time_utc": "2013-01-01T08:00:00Z",
         "air_temperature_c": 22, "dew_point_c": None, "distance_to_circuit_km": 30},
        {"race_id": "b", "observation_time_utc": "2014-01-01T08:00:00Z",
         "air_temperature_c": 25, "dew_point_c": 15, "distance_to_circuit_km": 31},
    ]
    before = deepcopy(observations)
    rows = compare_seasons(events, observations)
    assert len(rows) == 1
    assert rows[0]["2013_observation_count"] == 2
    assert rows[0]["2013_temperature_mean_c"] == 21
    assert rows[0]["2014_temperature_mean_c"] == 25
    assert rows[0]["temperature_mean_difference_2014_minus_2013_c"] == 4
    assert rows[0]["temperature_delta_scope"].endswith("descriptive only")
    assert observations == before
    assert parse_utc("2014-01-01T08:00:00Z").tzinfo is not None


def test_observation_sort_audit_is_per_event_and_includes_report_type():
    rows = [
        {"race_id": "a", "station_id": "x", "observation_time_utc": "2013-01-01T08:00:00Z", "report_type": "FM-15"},
        {"race_id": "a", "station_id": "x", "observation_time_utc": "2013-01-01T08:00:00Z", "report_type": "SY-MT"},
        {"race_id": "b", "station_id": "y", "observation_time_utc": "2014-01-01T08:00:00Z", "report_type": "FM-12"},
    ]
    assert audit_observation_rows(rows)["sorted_within_each_event"] is True
    unsorted = [rows[1], rows[0], rows[2]]
    assert audit_observation_rows(unsorted)["sorted_within_each_event"] is False
    assert audit_observation_rows(rows)["record_count"] == 3


def test_common_report_type_comparison_excludes_noncommon_types():
    events = [
        {"season": "2013", "grand_prix": "GP", "race_id": "a"},
        {"season": "2014", "grand_prix": "GP", "race_id": "b"},
    ]
    observations = [
        {"race_id": "a", "report_type": "FM-15", "air_temperature_c": 20,
         "observation_time_utc": "2013-01-01T08:00:00Z"},
        {"race_id": "a", "report_type": "SY-MT", "air_temperature_c": 30,
         "observation_time_utc": "2013-01-01T09:00:00Z"},
        {"race_id": "b", "report_type": "FM-15", "air_temperature_c": 24,
         "observation_time_utc": "2014-01-01T10:00:00Z"},
        {"race_id": "b", "report_type": "FM-12", "air_temperature_c": 40,
         "observation_time_utc": "2014-01-01T11:00:00Z"},
    ]
    result = compare_common_report_types(events, observations)
    assert len(result) == 1
    assert result[0]["report_type"] == "FM-15"
    assert result[0]["temperature_mean_difference_2014_minus_2013_c"] == 4
    assert result[0]["2013_report_count"] == result[0]["2014_report_count"] == 1
    assert result[0]["2013_first_observation_utc"] != result[0]["2014_first_observation_utc"]
    no_common = compare_common_report_types(
        events,
        [{"race_id": "a", "report_type": "SY-MT"},
         {"race_id": "b", "report_type": "FM-12"}],
    )
    assert no_common == []


def test_event_evidence_and_instrumental_observations_are_separate_tables():
    with (CLIMATE_DATA / "race_evidence.csv").open(encoding="utf-8-sig", newline="") as handle:
        evidence_columns = set(next(csv.reader(handle)))
    assert {"session", "source_class", "source_url", "confidence", "limitation"} <= evidence_columns
    assert "observation_time_utc" not in evidence_columns
    assert "air_temperature_c" not in evidence_columns
    station_columns = set(read_csv(CLIMATE_DATA / "noaa_stations.csv")[0])
    assert {"station_id", "station_latitude", "station_longitude", "station_elevation_m"} <= station_columns


def test_phase3b_quality_classification_matches_stored_flags_and_counts():
    events = read_csv(CLIMATE_DATA / "events.csv")
    observations = read_csv(CLIMATE_DATA / "noaa_observations.csv")
    classified = classify_observation_quality(observations)
    assert len(events) == 10
    assert len(classified) == 90
    assert sum(row["tmp_quality_flag"] == "1" for row in classified) == 90
    assert sum(row["dew_quality_flag"] == "1" for row in classified) == 89
    assert sum(row["dew_quality_flag"] == "2" for row in classified) == 1
    assert all(parse_isd_measurement(row["tmp_raw"])["quality_flag"] == row["tmp_quality_flag"]
               for row in classified)
    assert all(parse_isd_measurement(row["dew_raw"])["quality_flag"] == row["dew_quality_flag"]
               for row in classified)
    assert sum(row["tmp_qc_eligible"] for row in classified) == 90
    assert sum(row["dew_qc_eligible"] for row in classified) == 89
    per_event = quality_summary_by_event(events, classified)
    assert len(per_event) == 10
    assert sum(row["observation_count"] for row in per_event) == len(classified)


def test_classification_does_not_modify_phase3b_data_or_snapshots():
    paths = sorted(path for path in CLIMATE_DATA.rglob("*") if path.is_file())
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    events = read_csv(CLIMATE_DATA / "events.csv")
    observations = read_csv(CLIMATE_DATA / "noaa_observations.csv")
    classified = classify_observation_quality(observations)
    quality_summary_by_event(events, classified)
    quality_summary_by_common_report_types(events, classified)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before


def test_sha256_manifest_records_current_state_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source.csv"
    source.write_bytes(b"a,b\\n1,2\\n")
    manifest = build_sha256_manifest([source, source], tmp_path, "2026-10-22T23:00:00Z")
    assert manifest["reference_point"] == "current state after Phase 3B; created for Phase 3C"
    assert manifest["historical_integrity_claim"] is False
    assert manifest["self_hash_excluded"] is True
    assert manifest["files"] == [{
        "path": "source.csv", "size_bytes": len(b"a,b\\n1,2\\n"),
        "sha256": hashlib.sha256(b"a,b\\n1,2\\n").hexdigest(),
    }]
    target = tmp_path / "manifest.json"
    target.write_text("sentinel", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_json_new(target, manifest)
    assert target.read_text(encoding="utf-8") == "sentinel"


def test_functions_do_not_modify_protected_raw_files():
    paths = sorted((ROOT / "data" / "raw").glob("*.csv"))
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    events = read_csv(CLIMATE_DATA / "events.csv")
    validate_events(events)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before
