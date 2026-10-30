"""Compara outputs e arquivos preservados: python -m src.verify_baseline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare_baseline(root: Path, baseline_path: Path) -> dict:
    root = Path(root)
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    metrics = json.loads((root / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    files = {}
    for name, expected in baseline["sha256"].items():
        path = root / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        files[name] = {"before_sha256": expected, "after_sha256": actual, "unchanged": actual == expected}
    differences = {key: {"before": expected, "after": metrics.get(key)}
                   for key, expected in baseline["metrics"].items() if metrics.get(key) != expected}
    return {"passed": not differences and all(item["unchanged"] for item in files.values()),
            "metrics_unchanged": not differences, "metric_differences": differences,
            "files": files,
            "scope": "CSV originais, app, notebook, modelos, código de gráficos e outputs numéricos. PNGs não comparados por hash; nenhum algoritmo de ML alterado.",
            "metrics_before": baseline["metrics"],
            "metrics_after": {key: metrics.get(key) for key in baseline["metrics"]}}


def main() -> int:
    result = compare_baseline(ROOT, ROOT / "outputs" / "phase_0_1_baseline.json")
    output = ROOT / "outputs" / "phase_0_1_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "metrics_unchanged": result["metrics_unchanged"],
                      "changed_files": [name for name, item in result["files"].items() if not item["unchanged"]],
                      "metric_differences": result["metric_differences"]}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
