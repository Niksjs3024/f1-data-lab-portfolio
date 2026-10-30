"""Rastreabilidade exercitada somente em arquivos temporários sintéticos."""
import hashlib
import json

import pytest

from src.verify_baseline import compare_baseline


@pytest.mark.parametrize("change", [None, "raw", "metrics", "missing"])
def test_baseline_detects_changes_and_missing_files(tmp_path, change):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    source = raw / "sample.csv"
    source.write_bytes(b"id,value\n1,10\n")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    metrics = {"records": 1, "rate": 50.0}
    (outputs / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    baseline = {"sha256": {"data/raw/sample.csv": hashlib.sha256(source.read_bytes()).hexdigest()}, "metrics": metrics}
    baseline_path = outputs / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    if change == "raw":
        source.write_bytes(b"id,value\n1,11\n")
    elif change == "metrics":
        (outputs / "metrics.json").write_text(json.dumps({"records": 2, "rate": 50.0}), encoding="utf-8")
    elif change == "missing":
        source.unlink()
    result = compare_baseline(tmp_path, baseline_path)
    assert result["passed"] == (change is None)
    assert result["metrics_unchanged"] == (change != "metrics")
    assert result["files"]["data/raw/sample.csv"]["unchanged"] == (change not in {"raw", "missing"})
    json.dumps(result, allow_nan=False)
