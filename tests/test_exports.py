"""Exportação estrita sobre dados controlados; nenhum output histórico é usado."""
import json

import numpy as np
import pandas as pd

from src import run_analysis
from src.data_quality import audit_tables, json_safe


def test_unknown_points_export_as_json_null_without_changing_denominator(synthetic_tables, tmp_path, monkeypatch):
    synthetic_tables["results"]["points"] = synthetic_tables["results"]["points"].astype(float)
    synthetic_tables["results"].loc[0, "points"] = float("nan")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    for name, table in synthetic_tables.items():
        table.to_csv(raw / f"{name}.csv", index=False)
    before = {path.name: path.read_bytes() for path in raw.iterdir()}
    monkeypatch.setattr(run_analysis, "ROOT", tmp_path)
    monkeypatch.setattr(run_analysis, "evaluate_models", lambda frame: {"synthetic_model_stub": True})
    monkeypatch.setattr(run_analysis, "save_figures", lambda *args: None)
    run_analysis.main()

    def reject_nonstandard_number(value):
        raise AssertionError(f"JSON fora do padrão: {value}")

    metrics_path = tmp_path / "outputs" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"), parse_constant=reject_nonstandard_number)
    assert metrics["grid_1"]["corridas"] == 2
    assert metrics["grid_1"]["vitoria_pct"] == 50.0
    assert metrics["grid_1"]["podio_pct"] == 100.0
    assert metrics["grid_1"]["pontos_pct"] is None
    probabilities = pd.read_csv(tmp_path / "outputs" / "probabilidades_por_grid.csv")
    assert pd.isna(probabilities.set_index("grid").loc[1, "pontos_pct"])
    assert before == {path.name: path.read_bytes() for path in raw.iterdir()}
    assert b"\r\n" not in metrics_path.read_bytes()
    assert b"\r\n" not in (tmp_path / "outputs" / "probabilidades_por_grid.csv").read_bytes()


def test_audit_json_preserves_infinite_invalid_value_as_evidence(synthetic_tables):
    synthetic_tables["results"]["grid"] = synthetic_tables["results"]["grid"].astype(float)
    synthetic_tables["results"].loc[0, "grid"] = float("inf")
    report = audit_tables(synthetic_tables)
    json.dumps(report, allow_nan=False)
    assert any(example.get("grid") == "Infinity" for issue in report["issues"] for example in issue["examples"])


def test_json_safe_distinguishes_unknown_and_invalid_infinite_values():
    result = json_safe({"unknown": np.nan, "count": np.int64(2), "invalid": np.inf, "missing": pd.NA})
    assert result == {"unknown": None, "count": 2, "invalid": "Infinity", "missing": None}
    json.dumps(result, allow_nan=False)
