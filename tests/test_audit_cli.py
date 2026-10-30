"""Auditoria/CLI exercitadas exclusivamente sobre CSVs sintéticos em tmp_path."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from src import audit_data


@pytest.fixture
def write_synthetic_raw(tmp_path):
    def write_tables(tables):
        raw = tmp_path / "synthetic_raw"
        raw.mkdir()
        for name, table in tables.items():
            table.to_csv(raw / f"{name}.csv", index=False, encoding="utf-8")
        return raw

    return write_tables


def source_bytes(raw):
    return {path.name: path.read_bytes() for path in raw.iterdir() if path.is_file()}


def read_written_report(output):
    report = json.loads((output / "data_quality.json").read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    markdown = (output / "data_quality.md").read_text(encoding="utf-8")
    assert markdown.strip()
    for inventory in report["files"].values():
        assert inventory["sha256"] in markdown
    return report


def test_create_audit_report_hashes_real_bytes_without_modifying_sources(
    synthetic_tables, write_synthetic_raw,
):
    raw = write_synthetic_raw(synthetic_tables)
    before = source_bytes(raw)
    report = audit_data.create_audit_report(raw)
    json.dumps(report, allow_nan=False)
    assert set(report["files"]) == set(synthetic_tables)
    for name, table in synthetic_tables.items():
        inventory = report["files"][name]
        content = before[f"{name}.csv"]
        assert inventory["file"] == f"{name}.csv"
        assert inventory["sha256"] == hashlib.sha256(content).hexdigest()
        assert inventory["bytes"] == len(content)
        assert inventory["csv_rows"] == len(table)
        assert inventory["header_fields"] == len(table.columns)
        assert report["tables"][name]["rows"] == len(table)
    assert source_bytes(raw) == before
    assert set(path.name for path in raw.iterdir()) == set(before)


def test_create_audit_report_distinguishes_calendar_result_and_dimension_coverage(
    synthetic_tables, write_synthetic_raw,
):
    synthetic_tables["races"] = pd.concat([
        synthetic_tables["races"],
        pd.DataFrame({"raceId": [12], "year": [2017], "circuitId": [22], "name": ["Sem resultados"]}),
    ], ignore_index=True)
    # Nascimento do piloto não pode ser usado como cobertura de resultados.
    synthetic_tables["drivers"]["dob"] = ["1960-01-01", "1970-02-02", "1980-03-03"]
    raw = write_synthetic_raw(synthetic_tables)
    report = audit_data.create_audit_report(raw)
    coverage = report["coverage"]
    assert coverage["calendar_years"] == {"min": 2014, "max": 2017}
    assert coverage["result_years"] == {"min": 2014, "max": 2015}
    assert coverage["calendar_by_year"] == {"2014": 1, "2015": 1, "2017": 1}
    assert coverage["results_by_year"] == {"2014": 3, "2015": 3}
    assert [race["raceId"] for race in coverage["races_without_results"]] == [12]
    for table in ("results", "drivers", "constructors", "status"):
        assert report["tables"][table]["associated_result_years"] == {"min": 2014, "max": 2015}


def test_create_audit_report_counts_custom_selection_and_explicit_exclusions(
    synthetic_tables, write_synthetic_raw, set_cell,
):
    synthetic_tables["races"]["year"] = [2003, 2017]
    synthetic_tables["results"].loc[3, "grid"] = 0
    set_cell(synthetic_tables, "results", "points", None, row=4)
    raw = write_synthetic_raw(synthetic_tables)
    report = audit_data.create_audit_report(raw, start_year=2017, end_year=2017)
    assert report["scope"] == {"start_year": 2017, "end_year": 2017}
    expected = {
        "input_results": 6, "outside_period": 3, "within_period": 3,
        "grid_zero_excluded": 1, "eligible_results": 2, "eligible_races": 1,
        "unknown_points": 1,
    }
    for key, count in expected.items():
        assert report["selection"][key] == count
    assert report["selection"]["grid_counts"] == [
        {"grid": 2, "records": 1, "distinct_races": 1},
        {"grid": 3, "records": 1, "distinct_races": 1},
    ]


def test_create_audit_report_can_use_preloaded_tables_without_mutation_or_reloading(
    synthetic_tables, write_synthetic_raw, monkeypatch,
):
    raw = write_synthetic_raw(synthetic_tables)
    before_tables = deepcopy(synthetic_tables)
    before_bytes = source_bytes(raw)

    def unexpected_reload(*args, **kwargs):
        pytest.fail("Tabelas já fornecidas não devem ser recarregadas dos CSVs")

    monkeypatch.setattr(audit_data, "read_csv", unexpected_reload)
    report = audit_data.create_audit_report(raw, tables=synthetic_tables)
    assert report["selection"]["input_results"] == 6
    assert set(report["files"]) == set(synthetic_tables)
    for name, before in before_tables.items():
        pd.testing.assert_frame_equal(synthetic_tables[name], before)
    assert source_bytes(raw) == before_bytes


@pytest.mark.parametrize("scenario,strict,expected_exit", [
    ("valid", False, 0),
    ("duplicate_grid", False, 0),
    ("duplicate_grid", True, 1),
    ("orphan", False, 1),
    ("missing_table", False, 1),
])
def test_cli_writes_reports_even_when_validation_fails_without_touching_csvs(
    synthetic_tables, write_synthetic_raw, tmp_path, scenario, strict, expected_exit,
):
    if scenario == "duplicate_grid":
        synthetic_tables["results"].loc[0, "grid"] = 2
    elif scenario == "orphan":
        synthetic_tables["results"].loc[0, "driverId"] = 999999
    elif scenario == "missing_table":
        del synthetic_tables["status"]
    raw = write_synthetic_raw(synthetic_tables)
    before = source_bytes(raw)
    output = tmp_path / "separate_reports" / "audit"
    args = ["--raw-dir", str(raw), "--output-dir", str(output)]
    if strict:
        args.append("--strict")
    assert audit_data.main(args) == expected_exit
    report = read_written_report(output)
    if scenario == "duplicate_grid":
        assert any(issue["severity"] == "warning" and not issue["blocking"]
                   for issue in report["issues"])
        assert not any(issue["blocking"] for issue in report["issues"])
    elif expected_exit:
        assert any(issue["blocking"] for issue in report["issues"])
    assert source_bytes(raw) == before
    assert set(path.name for path in raw.iterdir()) == set(before)
    assert not list(output.glob("*.csv"))


@pytest.mark.parametrize("location", ["raw", "nested", "normalized_raw"])
def test_cli_rejects_output_inside_raw_before_reading_or_writing(
    synthetic_tables, write_synthetic_raw, monkeypatch, location,
):
    raw = write_synthetic_raw(synthetic_tables)
    before = source_bytes(raw)
    outputs = {"raw": raw, "nested": raw / "reports", "normalized_raw": raw / "reports" / ".."}

    def unexpected_audit(*args, **kwargs):
        pytest.fail("Saída dentro de raw deve ser recusada antes de iniciar a auditoria")

    monkeypatch.setattr(audit_data, "create_audit_report", unexpected_audit)
    with pytest.raises(SystemExit) as exc:
        audit_data.main(["--raw-dir", str(raw), "--output-dir", str(outputs[location])])
    assert exc.value.code == 2
    assert source_bytes(raw) == before
    assert set(path.name for path in raw.iterdir()) == set(before)


def test_module_cli_strict_returns_one_and_persists_reports(
    synthetic_tables, write_synthetic_raw, tmp_path,
):
    synthetic_tables["results"].loc[0, "grid"] = 2
    raw = write_synthetic_raw(synthetic_tables)
    before = source_bytes(raw)
    output = tmp_path / "subprocess_reports"
    process = subprocess.run(
        [sys.executable, "-B", "-m", "src.audit_data", "--strict",
         "--raw-dir", str(raw), "--output-dir", str(output)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    report = read_written_report(output)
    assert any(issue["severity"] == "warning" for issue in report["issues"])
    assert not any(issue["blocking"] for issue in report["issues"])
    assert source_bytes(raw) == before
