"""Contrato de auditoria/validação usando apenas tabelas sintéticas.

Os códigos e textos das issues não são fixados: o contrato público ainda está
em definição. As verificações se concentram em severidade e comportamento.
"""
import json
from copy import deepcopy

import pandas as pd
import pytest


PRIMARY_KEYS = [
    ("results", "resultId"), ("races", "raceId"),
    ("drivers", "driverId"), ("constructors", "constructorId"),
    ("status", "statusId"),
]
FOREIGN_KEYS = ["raceId", "driverId", "constructorId", "statusId"]


def assert_report(report):
    assert isinstance(report, dict)
    assert isinstance(report["issues"], list)
    # allow_nan=False detecta também NaN/Infinity escondidos em exemplos.
    json.loads(json.dumps(report, allow_nan=False))
    for issue in report["issues"]:
        assert {"code", "table", "severity", "count", "examples", "blocking"} <= issue.keys()
        assert isinstance(issue["code"], str)
        assert isinstance(issue["table"], str)
        assert issue["severity"] in {"error", "warning", "info"}
        assert type(issue["count"]) is int
        assert issue["count"] >= 0
        assert isinstance(issue["examples"], list)
        assert type(issue["blocking"]) is bool


def assert_blocking_issue(report, table):
    assert_report(report)
    assert any(
        issue["table"] == table and issue["severity"] == "error" and issue["blocking"]
        for issue in report["issues"]
    )


def test_valid_tables_return_serializable_reports(quality_api, synthetic_tables):
    assert issubclass(quality_api.DataValidationError, ValueError)
    for report in (quality_api.audit_tables(synthetic_tables),
                   quality_api.validate_tables(synthetic_tables),
                   quality_api.validate_tables(synthetic_tables, strict=True)):
        assert_report(report)
        assert not any(issue["blocking"] for issue in report["issues"])


@pytest.mark.parametrize("table,key", PRIMARY_KEYS)
@pytest.mark.parametrize("defect", ["duplicate", "null"])
def test_primary_keys_block_in_all_tables(
    quality_api, synthetic_tables, set_cell, table, key, defect,
):
    value = synthetic_tables[table].at[1, key] if defect == "duplicate" else None
    set_cell(synthetic_tables, table, key, value)
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), table)
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


@pytest.mark.parametrize("key", FOREIGN_KEYS)
@pytest.mark.parametrize("value", [999999, None], ids=["orphan", "null"])
def test_foreign_keys_to_each_dimension_are_blocking(
    quality_api, synthetic_tables, set_cell, key, value,
):
    set_cell(synthetic_tables, "results", key, value)
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), "results")
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


@pytest.mark.parametrize("table", [key[0] for key in PRIMARY_KEYS])
def test_audit_missing_tables_reports_without_raising(quality_api, synthetic_tables, table):
    del synthetic_tables[table]
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), table)
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


@pytest.mark.parametrize("table,column", PRIMARY_KEYS + [
    ("results", "raceId"), ("results", "driverId"),
    ("results", "constructorId"), ("results", "statusId"),
    ("results", "grid"), ("results", "positionOrder"),
    ("results", "points"), ("races", "year"),
])
def test_audit_missing_columns_reports_instead_of_keyerror(
    quality_api, synthetic_tables, table, column,
):
    synthetic_tables[table] = synthetic_tables[table].drop(columns=column)
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), table)
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


def test_audit_empty_mapping_returns_schema_report(quality_api):
    report = quality_api.audit_tables({})
    assert_report(report)
    assert any(issue["blocking"] for issue in report["issues"])


def test_duplicate_participation_is_blocking_despite_distinct_result_ids(
    quality_api, synthetic_tables,
):
    synthetic_tables["results"].loc[1, "driverId"] = 1
    assert synthetic_tables["results"]["resultId"].is_unique
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), "results")
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


def test_participation_uniqueness_is_scoped_to_selected_modern_period(
    quality_api, synthetic_tables,
):
    synthetic_tables["races"].loc[0, "year"] = 1950
    synthetic_tables["results"].loc[1, "driverId"] = 1
    report = quality_api.validate_tables(synthetic_tables, start_year=2015, end_year=2015)
    assert_report(report)
    assert not any(issue["blocking"] for issue in report["issues"])


@pytest.mark.parametrize("key", FOREIGN_KEYS)
def test_foreign_key_integrity_is_checked_even_outside_selection(
    quality_api, synthetic_tables, key,
):
    synthetic_tables["races"].loc[0, "year"] = 1950
    synthetic_tables["results"].loc[0, key] = 999999
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables, start_year=2015, end_year=2015)


@pytest.mark.parametrize("table,key", PRIMARY_KEYS)
def test_primary_key_integrity_is_not_bypassed_by_year_filter(
    quality_api, synthetic_tables, table, key,
):
    # Nem mesmo uma seleção sem resultados deve ocultar dimensões/PKs inválidas.
    duplicate = synthetic_tables[table].iloc[[0]].copy()
    synthetic_tables[table] = pd.concat([synthetic_tables[table], duplicate], ignore_index=True)
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables, start_year=2017, end_year=2017)


@pytest.mark.parametrize("column", ["grid", "positionOrder"])
@pytest.mark.parametrize("value", [None, float("nan"), "inválido", float("inf"),
                                   float("-inf"), -1, 1.5])
def test_selected_grid_and_finishing_order_require_finite_nonnegative_integers(
    quality_api, synthetic_tables, set_cell, column, value,
):
    set_cell(synthetic_tables, "results", column, value)
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), "results")
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


@pytest.mark.parametrize("value", ["inválido", -0.5, float("inf"), float("-inf")])
def test_present_points_must_be_numeric_finite_and_nonnegative(
    quality_api, synthetic_tables, set_cell, value,
):
    set_cell(synthetic_tables, "results", "points", value)
    assert_blocking_issue(quality_api.audit_tables(synthetic_tables), "results")
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


@pytest.mark.parametrize("value", [None, float("nan"), pd.NA, "", " ", "\\N", "NULL"])
def test_unknown_points_are_not_blocking(quality_api, synthetic_tables, set_cell, value):
    set_cell(synthetic_tables, "results", "points", value)
    report = quality_api.validate_tables(synthetic_tables)
    assert_report(report)
    assert not any(issue["blocking"] for issue in report["issues"])


@pytest.mark.parametrize("value", ["inválido", -1, 1.25, float("inf"), float("-inf")])
def test_present_laps_must_be_finite_nonnegative_integers(
    quality_api, synthetic_tables, set_cell, value,
):
    set_cell(synthetic_tables, "results", "laps", value)
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)


def test_missing_laps_are_allowed_with_warning(quality_api, synthetic_tables, set_cell):
    baseline = quality_api.audit_tables(synthetic_tables)
    set_cell(synthetic_tables, "results", "laps", None)
    report = quality_api.validate_tables(synthetic_tables)
    assert_report(report)
    assert not any(issue["blocking"] for issue in report["issues"])
    existing_codes = {issue["code"] for issue in baseline["issues"]}
    assert any(issue["severity"] == "warning" and issue["table"] == "results"
               and issue["code"] not in existing_codes and issue["count"] == 1
               for issue in report["issues"])


def test_duplicate_positive_grid_is_nonblocking_warning_except_in_strict_mode(
    quality_api, synthetic_tables,
):
    synthetic_tables["results"].loc[0, "grid"] = 2
    report = quality_api.validate_tables(synthetic_tables, strict=False)
    assert_report(report)
    assert any(issue["table"] == "results" and issue["severity"] == "warning"
               and not issue["blocking"] for issue in report["issues"])
    assert not any(issue["blocking"] for issue in report["issues"])
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables, strict=True)


def test_grid_zero_is_counted_only_in_selected_period(quality_api, synthetic_tables):
    baseline = quality_api.audit_tables(synthetic_tables, start_year=2014, end_year=2014)
    # Dois zeros selecionados e um fora do período; zero repetido não é grid positivo duplicado.
    synthetic_tables["results"].loc[[0, 1, 3], "grid"] = 0
    report = quality_api.validate_tables(synthetic_tables, start_year=2014, end_year=2014)
    assert_report(report)
    assert not any(issue["blocking"] for issue in report["issues"])
    existing_codes = {issue["code"] for issue in baseline["issues"]}
    assert any(issue["table"] == "results" and issue["code"] not in existing_codes
               and issue["severity"] in {"info", "warning"} and issue["count"] == 2
               for issue in report["issues"])


@pytest.mark.parametrize("corrupt", [False, True], ids=["valid", "invalid"])
def test_audit_is_non_destructive_even_when_reporting_errors(
    quality_api, synthetic_tables, set_cell, corrupt,
):
    if corrupt:
        set_cell(synthetic_tables, "results", "points", float("inf"))
        set_cell(synthetic_tables, "drivers", "driverId", None)
    before = deepcopy(synthetic_tables)
    report = quality_api.audit_tables(synthetic_tables)
    assert_report(report)
    assert synthetic_tables.keys() == before.keys()
    for table in before:
        pd.testing.assert_frame_equal(synthetic_tables[table], before[table])


def test_validation_failure_does_not_mutate_inputs(quality_api, synthetic_tables, set_cell):
    set_cell(synthetic_tables, "results", "grid", "inválido")
    before = deepcopy(synthetic_tables)
    with pytest.raises(quality_api.DataValidationError):
        quality_api.validate_tables(synthetic_tables)
    for table in before:
        pd.testing.assert_frame_equal(synthetic_tables[table], before[table])
