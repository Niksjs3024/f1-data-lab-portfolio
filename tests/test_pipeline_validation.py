"""Regressões sintéticas da integração, preservação e probabilidades do pipeline."""
import json
import math
from copy import deepcopy

import pandas as pd
import pytest

from src.analysis import grid_probabilities
from src.pipeline import build_dataset, parse_lap_seconds


EXISTING_COLUMNS = {
    "result_id", "race_id", "driver_id", "constructor_id", "status_id",
    "grid", "finish_position", "points", "fastest_lap_time", "fastest_lap_speed",
    "year", "circuit_id", "grand_prix", "forename", "surname", "constructor",
    "status", "driver",
}
PROBABILITY_COLUMNS = [
    "grid", "corridas", "vitoria_pct", "podio_pct", "top10_pct", "pontos_pct",
]
PRIMARY_KEYS = [
    ("results", "resultId"), ("races", "raceId"), ("drivers", "driverId"),
    ("constructors", "constructorId"), ("status", "statusId"),
]


def test_build_preserves_columns_and_adds_quality_and_official_result_fields(synthetic_tables):
    df = build_dataset(synthetic_tables)
    assert EXISTING_COLUMNS | {"laps", "position", "position_text", "points_known"} <= set(df.columns)
    assert isinstance(df.attrs["data_quality"], dict)
    assert isinstance(df.attrs["data_quality"]["issues"], list)
    json.dumps(df.attrs["data_quality"], allow_nan=False)
    by_id = df.set_index("result_id")
    assert by_id.loc[101, "position"] == "1"
    assert by_id.loc[103, "position_text"] == "R"
    assert pd.isna(by_id.loc[103, "position"])
    assert by_id.loc[103, "finish_position"] == 19
    assert by_id.loc[103, "laps"] == 12
    assert df["points_known"].eq(True).all()


def test_valid_joins_never_multiply_or_lose_results(synthetic_tables):
    df = build_dataset(synthetic_tables)
    assert len(df) == len(synthetic_tables["results"]) == 6
    assert df["result_id"].is_unique
    assert set(df["result_id"]) == set(synthetic_tables["results"]["resultId"])
    assert df.groupby("race_id").size().to_dict() == {10: 3, 11: 3}
    by_id = df.set_index("result_id")
    assert by_id.loc[101, "driver"] == "Ana Um"
    assert by_id.loc[103, "constructor"] == "Equipe B"
    assert by_id.loc[106, "grand_prix"] == "GP B"
    assert by_id.loc[106, "status"] == "Engine"


@pytest.mark.parametrize("table,key", PRIMARY_KEYS)
@pytest.mark.parametrize("defect", ["duplicate", "null"])
def test_bad_primary_keys_fail_before_any_join(
    quality_api, synthetic_tables, set_cell, monkeypatch, table, key, defect,
):
    value = synthetic_tables[table].at[1, key] if defect == "duplicate" else None
    set_cell(synthetic_tables, table, key, value)

    def unexpected_join(*args, **kwargs):
        pytest.fail("Uma junção foi iniciada antes da rejeição das PKs inválidas")

    monkeypatch.setattr(pd.DataFrame, "merge", unexpected_join)
    monkeypatch.setattr(pd, "merge", unexpected_join)
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables)


@pytest.mark.parametrize("key", ["raceId", "driverId", "constructorId", "statusId"])
@pytest.mark.parametrize("value", [999999, None], ids=["orphan", "null"])
def test_bad_foreign_keys_fail_before_any_join(
    quality_api, synthetic_tables, set_cell, monkeypatch, key, value,
):
    set_cell(synthetic_tables, "results", key, value)

    def unexpected_join(*args, **kwargs):
        pytest.fail("Uma junção foi iniciada antes da rejeição das FKs inválidas")

    monkeypatch.setattr(pd.DataFrame, "merge", unexpected_join)
    monkeypatch.setattr(pd, "merge", unexpected_join)
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables)


def test_build_rejects_duplicate_participation_with_unique_result_ids(quality_api, synthetic_tables):
    synthetic_tables["results"].loc[1, "driverId"] = 1
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables)


def test_duplicate_grid_keeps_all_rows_in_compatibility_and_rejects_strict(
    quality_api, synthetic_tables,
):
    synthetic_tables["results"].loc[0, "grid"] = 2
    df = build_dataset(synthetic_tables, strict=False)
    assert len(df) == 6
    assert df["result_id"].is_unique
    assert any(issue["severity"] == "warning"
               for issue in df.attrs["data_quality"]["issues"])
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables, strict=True)


@pytest.mark.parametrize("column", ["grid", "positionOrder"])
@pytest.mark.parametrize("value", [None, float("nan"), "inválido", -1, 1.5,
                                   float("inf"), float("-inf")])
def test_build_does_not_silently_drop_or_coerce_invalid_selected_results(
    quality_api, synthetic_tables, set_cell, column, value,
):
    set_cell(synthetic_tables, "results", column, value)
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables)


@pytest.mark.parametrize("column", ["grid", "positionOrder"])
def test_grid_and_finish_validation_respects_custom_selected_years(
    synthetic_tables, set_cell, column,
):
    set_cell(synthetic_tables, "results", column, "histórico sem informação")
    df = build_dataset(synthetic_tables, start_year=2015, end_year=2015)
    assert set(df["result_id"]) == {104, 105, 106}


def test_year_limits_are_inclusive_and_customizable(synthetic_tables):
    synthetic_tables["races"]["year"] = [2004, 2017]
    assert set(build_dataset(synthetic_tables)["result_id"]) == {101, 102, 103, 104, 105, 106}
    assert set(build_dataset(synthetic_tables, 2004, 2004)["result_id"]) == {101, 102, 103}
    assert set(build_dataset(synthetic_tables, 2017, 2017)["result_id"]) == {104, 105, 106}
    synthetic_tables["races"]["year"] = [2003, 2018]
    assert build_dataset(synthetic_tables).empty


def test_zero_grid_is_explicitly_excluded_without_removing_other_rows(synthetic_tables):
    synthetic_tables["results"].loc[[0, 1, 3], "grid"] = 0
    df = build_dataset(synthetic_tables, start_year=2014, end_year=2014)
    assert df["result_id"].tolist() == [103]
    assert df["grid"].gt(0).all()
    assert any(issue["table"] == "results" and issue["count"] == 2
               and issue["severity"] in {"info", "warning"}
               for issue in df.attrs["data_quality"]["issues"])


@pytest.mark.parametrize("status_id,status_text,position_text", [
    (2, "Engine", "R"), (3, "Disqualified", "D"), (4, "Withdrawn", "W"),
])
def test_retirements_disqualifications_and_withdrawals_remain_in_denominator(
    synthetic_tables, status_id, status_text, position_text,
):
    results = synthetic_tables["results"]
    results.loc[2, "statusId"] = status_id
    results.loc[2, "positionText"] = position_text
    results.loc[2, "laps"] = 0
    results.loc[2, ["fastestLapTime", "fastestLapSpeed"]] = None
    df = build_dataset(synthetic_tables)
    row = df.set_index("result_id").loc[103]
    assert row["status"] == status_text
    assert row["position_text"] == position_text
    assert pd.isna(row["position"])
    assert row["finish_position"] == 19
    assert row["laps"] == 0
    assert len(df) == 6
    summary = grid_probabilities(df).set_index("grid")
    assert summary.loc[3, "corridas"] == 2
    assert summary.loc[3, "pontos_pct"] == 0


@pytest.mark.parametrize("lap", [None, "\\N", "inválido", "-1:20", "1:60", "1:inf"])
def test_missing_or_invalid_fastest_lap_never_removes_result(synthetic_tables, set_cell, lap):
    set_cell(synthetic_tables, "results", "fastestLapTime", lap, row=2)
    set_cell(synthetic_tables, "results", "fastestLapSpeed", None, row=2)
    df = build_dataset(synthetic_tables)
    assert len(df) == 6
    assert 103 in set(df["result_id"])
    assert df.set_index("result_id").loc[103, "status"] == "Engine"
    assert grid_probabilities(df).set_index("grid").loc[3, "corridas"] == 2


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, "", " ", "\\N", "NULL"])
def test_unknown_points_stay_missing_and_keep_group_denominator(synthetic_tables, set_cell, value):
    set_cell(synthetic_tables, "results", "points", value)
    df = build_dataset(synthetic_tables)
    row = df.set_index("result_id").loc[101]
    assert pd.isna(row["points"])
    assert not bool(row["points_known"])
    assert df["points_known"].sum() == 5
    summary = grid_probabilities(df)
    assert summary.columns.tolist() == PROBABILITY_COLUMNS
    by_grid = summary.set_index("grid")
    assert by_grid.loc[1, "corridas"] == 2
    assert math.isnan(by_grid.loc[1, "pontos_pct"])
    assert by_grid.loc[1, "vitoria_pct"] == 50
    assert by_grid.loc[1, "podio_pct"] == 100
    assert by_grid.loc[1, "top10_pct"] == 100
    assert by_grid.loc[2, "corridas"] == 2
    assert by_grid.loc[2, "pontos_pct"] == 100
    assert by_grid.loc[3, "pontos_pct"] == 0


@pytest.mark.parametrize("value", ["inválido", -1, float("inf"), float("-inf")])
def test_invalid_present_points_cannot_become_unknown_or_zero(
    quality_api, synthetic_tables, set_cell, value,
):
    set_cell(synthetic_tables, "results", "points", value)
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables)


def test_zero_and_fractional_points_are_known_values(synthetic_tables):
    synthetic_tables["results"]["points"] = [0, 0.5, 0, 18, 25, 0]
    df = build_dataset(synthetic_tables)
    assert df["points_known"].eq(True).all()
    by_grid = grid_probabilities(df).set_index("grid")
    assert by_grid.loc[1, "pontos_pct"] == 50
    assert by_grid.loc[2, "pontos_pct"] == 100


@pytest.mark.parametrize("with_known_flag", [False, True], ids=["legacy", "points_known"])
def test_probabilities_keep_six_columns_and_original_group_sizes_with_unknown_points(with_known_flag):
    df = pd.DataFrame({
        "result_id": [1, 2, 3, 4, 5], "grid": [1, 1, 2, 2, 2],
        "finish_position": [1, 11, 1, 2, 10], "points": [25, None, 25, 18, 0],
    })
    if with_known_flag:
        df["points_known"] = df["points"].notna()
    before = deepcopy(df)
    summary = grid_probabilities(df)
    assert summary.columns.tolist() == PROBABILITY_COLUMNS
    by_grid = summary.set_index("grid")
    assert by_grid["corridas"].to_dict() == {1: 2, 2: 3}
    assert pd.isna(by_grid.loc[1, "pontos_pct"])
    assert by_grid.loc[1, "vitoria_pct"] == 50
    assert by_grid.loc[1, "podio_pct"] == 50
    assert by_grid.loc[1, "top10_pct"] == 50
    assert by_grid.loc[2, "pontos_pct"] == 66.67
    assert by_grid.loc[2, "vitoria_pct"] == 33.33
    assert by_grid.loc[2, "podio_pct"] == 66.67
    assert by_grid.loc[2, "top10_pct"] == 100
    pd.testing.assert_frame_equal(df, before)


def test_all_unknown_points_do_not_erase_grid_group():
    df = pd.DataFrame({
        "result_id": [1, 2], "grid": [4, 4],
        "finish_position": [1, 12], "points": [None, None],
        "points_known": [False, False],
    })
    summary = grid_probabilities(df).set_index("grid")
    assert summary.loc[4, "corridas"] == 2
    assert pd.isna(summary.loc[4, "pontos_pct"])
    assert summary.loc[4, "vitoria_pct"] == 50
    assert summary.loc[4, "top10_pct"] == 50


@pytest.mark.parametrize("missing", [
    ["laps"], ["position"], ["positionText"], ["laps", "position", "positionText"],
])
def test_legacy_fixtures_can_omit_new_optional_source_columns(synthetic_tables, missing):
    synthetic_tables["results"] = synthetic_tables["results"].drop(columns=missing)
    df = build_dataset(synthetic_tables)
    assert len(df) == 6
    assert {"laps", "position", "position_text", "points_known"} <= set(df.columns)
    targets = {"laps": "laps", "position": "position", "positionText": "position_text"}
    for column in missing:
        assert df[targets[column]].isna().all()


def test_missing_laps_preserve_rows_and_are_reported(synthetic_tables, set_cell):
    set_cell(synthetic_tables, "results", "laps", None)
    df = build_dataset(synthetic_tables)
    assert len(df) == 6
    assert pd.isna(df.set_index("result_id").loc[101, "laps"])
    assert any(issue["table"] == "results" and issue["severity"] == "warning"
               for issue in df.attrs["data_quality"]["issues"])


@pytest.mark.parametrize("value", ["inválido", -1, 0.5, float("inf"), float("-inf")])
def test_build_rejects_invalid_laps(quality_api, synthetic_tables, set_cell, value):
    set_cell(synthetic_tables, "results", "laps", value)
    with pytest.raises(quality_api.DataValidationError):
        build_dataset(synthetic_tables)


def test_build_does_not_mutate_any_input_table(synthetic_tables):
    before = deepcopy(synthetic_tables)
    df = build_dataset(synthetic_tables)
    for table in before:
        pd.testing.assert_frame_equal(synthetic_tables[table], before[table])
    # A saída tampouco deve compartilhar dados mutáveis com os frames de origem.
    df.loc[0, "grid"] = 99
    for table in before:
        pd.testing.assert_frame_equal(synthetic_tables[table], before[table])


@pytest.mark.parametrize("value", [
    "-1:20", "1:-20", "-1:02:03", "1:-02:03", "1:02:-03",
    "1:60", "1:60.001", "1:02:60", "inf:20", "1:inf", "1:02:inf",
    "nan:20", "1:nan", float("inf"), float("-inf"), None, "\\N", "N/A", "inválido",
])
def test_lap_parser_rejects_negative_nonfinite_and_out_of_range_seconds(value):
    assert math.isnan(parse_lap_seconds(value))


@pytest.mark.parametrize("value,expected", [
    ("01:27.5", 87.5), ("1:02:03.2", 3723.2),
    ("0:00.001", 0.001), (" 1:59.999 ", 119.999),
])
def test_lap_parser_keeps_existing_valid_formats(value, expected):
    assert parse_lap_seconds(value) == pytest.approx(expected)
