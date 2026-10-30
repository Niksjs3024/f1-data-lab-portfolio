"""Execute: python -m src.run_analysis"""
import json
from pathlib import Path
from .pipeline import load_data, build_dataset
from .analysis import grid_probabilities, compare_2013_2014
from .model import evaluate_models, evaluate_models_with_arrays
from .model_figures import save_model_figures
from .visualization import save_figures
from .audit_data import create_audit_report, write_audit_report
from .data_quality import enforce_report, json_safe

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tables = load_data(ROOT / "data" / "raw")
    output = ROOT / "outputs"
    report = create_audit_report(ROOT / "data" / "raw", tables=tables)
    write_audit_report(report, output)
    enforce_report(report)
    df = build_dataset(tables)
    report["joins"] = df.attrs["data_quality"]["joins"]
    report["selection"]["output_results"] = int(len(df))
    write_audit_report(report, output)
    proba = grid_probabilities(df)
    paired, lap_stats = compare_2013_2014(df)
    model_stats, plot_arrays = evaluate_models_with_arrays(df)
    output.mkdir(exist_ok=True)
    proba.to_csv(output / "probabilidades_por_grid.csv", index=False, lineterminator="\n")
    paired.to_csv(output / "comparacao_2013_2014.csv", index=False, lineterminator="\n")
    save_figures(df, proba, paired, output / "figures")
    save_model_figures(model_stats, plot_arrays, output / "figures")
    metrics = {
        "escopo": "2004–2017; grid > 0; sem excluir resultados por voltas rápidas faltantes",
        "resultados_iniciais": int(len(tables["results"])),
        "resultados_validos_periodo": int(len(df)),
        "grid_1": proba.loc[proba.grid.eq(1)].iloc[0].to_dict(),
        "comparacao_2013_2014": lap_stats,
        "modelagem": model_stats,
    }
    metrics = json_safe(metrics)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
    print("\nArquivos exportados para:", output)


if __name__ == "__main__":
    main()
