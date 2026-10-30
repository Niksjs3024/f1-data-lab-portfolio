"""Contratos da camada histórica isolada, usando somente dados sintéticos."""
from pathlib import Path

import pandas as pd
import pytest

from src.historical_reconciliation import (
    build_comparisons,
    export_artifacts,
    extract_raw_records,
    load_references,
    reconcile_records,
    round_half_up,
)


def records_fixture():
    rows = []
    for year, time, speed, lap, rank, laps, status_id, gp in [
        (2013, "1:30.0", 210.0, 20, 4, 58, 1, "Japanese Grand Prix"),
        (2014, "1:32.0", 211.0, 19, 5, 57, 1, "Japanese Grand Prix"),
        (2013, "1:31.0", 209.0, 18, 6, 58, 1, "Australian Grand Prix"),
        (2014, "1:33.0", 208.0, 30, 9, 3, 2, "Australian Grand Prix"),
    ]:
        rows.append({"result_id": len(rows) + 1, "race_id": 100 + year, "year": year,
                     "grand_prix": gp, "event_key": "", "circuit_id": 1 if "Australian" in gp else 2,
                     "driver_id": 44, "constructor_id": 7, "driver": "Test Driver", "constructor": "Team",
                     "fastestLapTime": time, "fastestLapSpeed": speed, "fastestLap": lap,
                     "rank": rank, "laps": laps, "statusId": status_id, "status": "Engine" if status_id == 2 else "Finished"})
    return pd.DataFrame(rows)


def reference_row(year, gp, circuit, **changes):
    row = {"year": year, "grand_prix": gp, "circuit_id": circuit, "driver_id": 44,
           "verified_fields": "fastestLapTime;fastestLapSpeed;fastestLap;rank",
           "official_fastestLapTime": 92.25 if year == 2014 else 90.0,
           "official_fastestLapSpeed": 211.0 if year == 2014 else 210.0,
           "official_fastestLap": 19 if year == 2014 else 20,
           "official_rank": 5 if year == 2014 else 4,
           "official_laps": "", "official_statusId": "", "source": "F1 official result",
           "source_url": "https://example.org/official", "consulted_at_utc": "2026-10-22",
           "confidence": "high", "justification": "Synthetic fixture; no factual transcription."}
    row.update(changes)
    return row


def refs_fixture():
    return pd.DataFrame([reference_row(2013, "Japanese Grand Prix", 2),
                         reference_row(2014, "Japanese Grand Prix", 2),
                         reference_row(2013, "Australian Grand Prix", 1),
                         reference_row(2014, "Australian Grand Prix", 1)])


def test_field_level_reconciliation_preserves_raw_and_only_corrects_verified_fields():
    raw = records_fixture()
    before = raw.copy(deep=True)
    out, log = reconcile_records(raw, refs_fixture())
    assert raw.equals(before)
    assert out.loc[1, "fastestLapTime_original"] == "1:32.0"
    assert out.loc[1, "fastestLapTime_verified"] == 92.25
    assert out.loc[1, "fastestLapTime_verification_status"] == "divergente"
    assert out.loc[1, "fastestLapTime_reconciled"] == 92.25
    assert out.loc[1, "laps_verification_status"] == "nao_verificado"
    assert out.loc[1, "laps_reconciled"] == 57
    assert set(log["field"]) == {"fastestLapTime", "fastestLapSpeed", "fastestLap", "rank"}
    assert log["correction_id"].is_unique and log["applied"].all()
    assert out["early_lap_count_alert"].sum() == 1


def test_duplicate_reference_or_raw_event_driver_is_ambiguous_and_not_applied():
    raw = records_fixture()
    refs = refs_fixture()
    refs = pd.concat([refs, refs.iloc[[1]]], ignore_index=True)
    out, log = reconcile_records(raw, refs)
    assert out.loc[1, "fastestLapTime_verification_status"] == "ambiguo"
    assert out.loc[1, "fastestLapTime_reconciled"] == 92.0
    assert not ((log["year"] == 2014) & (log["grand_prix"] == "Japanese Grand Prix")).any()
    duplicate_row = raw.iloc[[0]].copy()
    duplicate_row.loc[:, "result_id"] = 99
    duplicated = pd.concat([raw, duplicate_row], ignore_index=True)
    out, _ = reconcile_records(duplicated, refs_fixture())
    assert out.iloc[0]["fastestLapTime_verification_status"] == "ambiguo"


def test_unverified_and_missing_values_are_not_imputed():
    raw = records_fixture()
    raw.loc[0, "fastestLapTime"] = None
    refs = refs_fixture().iloc[[1]].copy()
    out, _ = reconcile_records(raw, refs)
    row = out.iloc[0]
    assert row["fastestLapTime_original"] is None
    assert pd.isna(row["fastestLapTime_reconciled"])
    assert row["fastestLapTime_verification_status"] == "nao_verificado"
    assert row["laps_reconciled"] == 58


def test_same_pair_comparison_separates_correction_from_eligibility():
    reconciled, _ = reconcile_records(records_fixture(), refs_fixture())
    pairs, summary = build_comparisons(reconciled)
    assert len(pairs) == 2
    assert pairs["scenario_b_eligible"].sum() == 2
    assert summary["coverage"]["original_eligible_pairs"] == 2
    assert summary["coverage"]["scenario_b_fully_verified_pairs"] == 2
    assert summary["original_on_scenario_b_eligible_pairs"]["pairs"] == summary["scenario_b_on_same_scenario_b_eligible_pairs"]["pairs"] == 2
    assert summary["metric_differences_same_pairs"]["scenario_b_minus_original_common_mean_s"] == pytest.approx(0.25)
    assert summary["scenario_a_all_original_eligible_pairs"]["japan_excluded"]["pairs"] == 1
    assert summary["scenario_a_all_original_eligible_pairs"]["early_retirement_proxy_pairs"]["pairs"] == 1


def test_rounding_is_explicit_half_up_and_reference_schema_rejects_bad_fields():
    assert round_half_up(92.25, 1) == 92.3
    assert round_half_up(92.24, 1) == 92.2
    bad = refs_fixture().iloc[[0]].copy()
    bad.loc[:, "verified_fields"] = "fastestLapTime;weather"
    with pytest.raises(ValueError, match="sem suporte"):
        load_references(bad)


def test_raw_extraction_preserves_input_tables_and_result_cardinality(synthetic_tables):
    synthetic_tables["races"].loc[:, "year"] = [2013, 2014]
    before = {name: frame.copy(deep=True) for name, frame in synthetic_tables.items()}
    records = extract_raw_records(synthetic_tables)
    assert len(records) == len(synthetic_tables["results"])
    assert records["result_id"].is_unique
    assert set(records["year"].unique()) == {2013, 2014}
    for name, frame in before.items():
        pd.testing.assert_frame_equal(synthetic_tables[name], frame)


def test_export_requires_a_new_destination_and_preserves_existing_files(tmp_path):
    raw = records_fixture()
    rec, log = reconcile_records(raw, refs_fixture())
    pairs, summary = build_comparisons(rec)
    target = tmp_path / "new-output"
    export_artifacts(raw, rec, log, pairs, summary, {"source_record_count_2013_2014": len(raw)}, target)
    assert (target / "original_records_2013_2014.csv").exists()
    assert (target / "reconciliation_summary_2013_2014.json").exists()
    sentinel = target / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_artifacts(raw, rec, log, pairs, summary, {}, target)
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"
