"""Testes para os diagnósticos de modelagem da Fase 2 (dados sintéticos)."""
from copy import deepcopy

import json
import math

import numpy as np
import pandas as pd
import pytest

from src import run_analysis
from src.model import (
    evaluate_models,
    evaluate_models_with_arrays,
    classification_diagnostics,
    regression_diagnostics,
    select_threshold_train,
    temporal_validation_classification,
    temporal_validation_regression,
)
from src.model_figures import save_model_figures
from src.pipeline import build_dataset

EXPECTED_MODEL_PNGS = {
    "modelo_mae_comparacao.png",
    "modelo_observado_vs_previsto.png",
    "modelo_distribuicao_erros.png",
    "modelo_curva_roc.png",
    "modelo_curva_pr.png",
    "modelo_curva_calibracao.png",
    "modelo_validacao_temporal_treino.png",
}

TRAIN_YEARS = [2013, 2014]
TEST_YEARS = [2015, 2016]


def _tables_for_years(years):
    """Tabelas sintéticas com 3 participantes por ano, incluindo abandono."""
    results = []
    races = []
    drivers = [
        {"driverId": 1, "forename": "Ana", "surname": "Um"},
        {"driverId": 2, "forename": "Beto", "surname": "Dois"},
        {"driverId": 3, "forename": "Caio", "surname": "Três"},
    ]
    constructors = [{"constructorId": 7, "name": "Equipe A"}]
    status = [
        {"statusId": 1, "status": "Finished"},
        {"statusId": 2, "status": "Engine"},
    ]
    result_id = 1
    race_id = 1
    for year in years:
        races.append({"raceId": race_id, "year": year, "circuitId": year, "name": f"GP {year}"})
        for grid in (1, 2, 3):
            retired = grid == 3
            results.append(
                {
                    "resultId": result_id,
                    "raceId": race_id,
                    "year": year,
                    "driverId": grid,
                    "constructorId": 7,
                    "statusId": 2 if retired else 1,
                    "grid": grid,
                    "positionOrder": 19 if retired else grid,
                    "position": str(grid) if not retired else None,
                    "positionText": str(grid) if not retired else "R",
                    "points": [25, 18, 0][grid - 1],
                    "laps": 58 if not retired else 12,
                    "fastestLapTime": "1:25.0",
                    "fastestLapSpeed": 215,
                }
            )
            result_id += 1
        race_id += 1
    return {
        "results": pd.DataFrame(results),
        "races": pd.DataFrame(races),
        "drivers": pd.DataFrame(drivers),
        "constructors": pd.DataFrame(constructors),
        "status": pd.DataFrame(status),
    }


def _df_for_years(years):
    return build_dataset(_tables_for_years(years))


def test_evaluate_models_preserves_existing_keys_and_values():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    stats = evaluate_models(df)
    assert stats["treino_anos"] == "2004–2014" or stats["treino_anos"] == "2013–2014"
    assert stats["teste_anos"] == "2015–2017" or stats["teste_anos"] == "2015–2016"
    assert stats["amostras_treino"] == len(df[df["year"] <= 2014])
    assert stats["amostras_teste"] == len(df[df["year"] >= 2015])
    reg = stats["regressao_linear"]
    for key in ("mae", "r2", "rmse", "coeficiente_grid", "intercepto"):
        assert key in reg
    assert reg["mae"] is not None and reg["mae"] > 0
    assert reg["rmse"] is not None and reg["rmse"] > 0
    refs = stats["referencias"]
    assert refs["mae_previsao_posicao_igual_ao_grid"] is not None
    assert refs["mae_previsao_constante_mediana"] is not None
    clf = stats["classificacao_podio"]
    assert clf["roc_auc"] is not None
    assert clf["brier_score"] is not None
    assert clf["train_prevalence"] > 0


def test_regression_has_error_distribution_and_out_of_range_fields():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    reg = evaluate_models(df)["regressao_linear"]
    for key in (
        "mean_error",
        "median_error",
        "mean_abs_error",
        "std_error",
        "outside_valid_range_count",
        "outside_valid_range_pct",
        "valid_range_min",
        "valid_range_max",
    ):
        assert key in reg
    assert reg["valid_range_min"] == 1.0
    assert reg["outside_valid_range_pct"] is None or 0 <= reg["outside_valid_range_pct"] <= 100


def test_classifier_has_pr_auc_logloss_calibration_and_grid_ranking():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    clf = evaluate_models(df)["classificacao_podio"]
    for key in (
        "pr_auc",
        "log_loss",
        "calibration_ece",
        "calibration_mce",
        "calibration_bins",
        "grid_ranking_auc",
        "constant_roc_auc",
        "constant_pr_auc",
        "constant_log_loss",
        "constant_brier_score",
        "threshold_criterion",
        "erro_classificador",
    ):
        assert key in clf
    assert clf["threshold_criterion"] == "youden_no_treino"
    assert clf["erro_classificador"] is None
    assert len(clf["calibration_bins"]) == 10
    cm = clf["confusion_matrix"]
    assert sum(sum(row) for row in cm) == int(len(df[df["year"] >= 2015]))


def test_threshold_is_train_only_and_deterministic():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    train = df[df["year"] <= 2014]
    y_train = train["finish_position"].between(1, 3).astype(int)
    # Probabilidades arbitrárias fixas; o limiar depende só do treino.
    p_train = pd.Series([0.9, 0.8, 0.1, 0.9, 0.2, 0.05])
    t1 = select_threshold_train(y_train, p_train)
    t2 = select_threshold_train(y_train, p_train)
    assert 0 < t1 < 1
    assert t1 == t2


def test_constant_prevalence_baseline_is_computed_on_train_only():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    train = df[df["year"] <= 2014]
    expected = float((train["finish_position"].between(1, 3)).mean())
    clf = evaluate_models(df)["classificacao_podio"]
    assert abs(clf["train_prevalence"] - expected) < 1e-9
    # Prevalência constante no treino: AP ≈ prevalência, Brier = p(1-p).
    assert abs(clf["constant_pr_auc"] - expected) < 1e-6
    assert abs(clf["constant_brier_score"] - expected * (1 - expected)) < 1e-6


def test_constant_brier_score_usa_rotulos_avaliados_com_p_do_treino():
    """Brier do baseline constante usa rótulos do conjunto avaliado com p estimada no treino.

    Quando a prevalencia do treino difere da do teste,
    constant_brier_score != p(1-p).
    """

    def linha(rid, race_id, year, grid, pos, pts, status_id=1, laps=58, fl="1:25.0", spd=215):
        pos_txt = str(pos) if pos <= 10 else None
        return {
            "resultId": rid,
            "raceId": race_id,
            "year": year,
            "driverId": grid,
            "constructorId": 7,
            "statusId": status_id,
            "grid": grid,
            "positionOrder": pos,
            "position": pos_txt,
            "positionText": pos_txt if pos_txt else "R",
            "points": pts,
            "laps": laps,
            "fastestLapTime": fl,
            "fastestLapSpeed": spd,
        }

    resultados = []
    rid = 1
    # 2013: 2 pódios, 1 não-pódio
    for g, pos, pts, sid, laps in ((1, 1, 25, 1, 58), (2, 2, 18, 1, 58), (3, 19, 0, 2, 12)):
        resultados.append(linha(rid, 1, 2013, g, pos, pts, sid, laps)); rid += 1
    # 2014: 0 pódios, 3 não-pódios
    for g, pos, pts in ((1, 19, 0), (2, 19, 0), (3, 19, 0)):
        resultados.append(linha(rid, 2, 2014, g, pos, pts, 2, 10)); rid += 1
    # 2015 e 2016: 3 pódios cada
    for race_id, year in ((3, 2015), (4, 2016)):
        for g, pos, pts in ((1, 1, 25), (2, 2, 18), (3, 3, 15)):
            resultados.append(linha(rid, race_id, year, g, pos, pts)); rid += 1

    tables = {
        "results": pd.DataFrame(resultados),
        "races": pd.DataFrame([
            {"raceId": 1, "year": 2013, "circuitId": 1, "name": "GP 2013"},
            {"raceId": 2, "year": 2014, "circuitId": 1, "name": "GP 2014"},
            {"raceId": 3, "year": 2015, "circuitId": 1, "name": "GP 2015"},
            {"raceId": 4, "year": 2016, "circuitId": 1, "name": "GP 2016"},
        ]),
        "drivers": pd.DataFrame([
            {"driverId": 1, "forename": "A", "surname": "U"},
            {"driverId": 2, "forename": "B", "surname": "D"},
            {"driverId": 3, "forename": "C", "surname": "T"},
        ]),
        "constructors": pd.DataFrame([{"constructorId": 7, "name": "Eq"}]),
        "status": pd.DataFrame([
            {"statusId": 1, "status": "Finished"},
            {"statusId": 2, "status": "Engine"},
        ]),
    }
    df = build_dataset(tables)
    train = df[df["year"] <= 2014]
    test = df[df["year"] >= 2015]
    train_prev = float((train["finish_position"].between(1, 3)).mean())
    test_prev = float((test["finish_position"].between(1, 3)).mean())
    assert abs(train_prev - 1 / 3) < 1e-9
    assert abs(test_prev - 1.0) < 1e-9
    assert train_prev != test_prev

    clf = evaluate_models(df)["classificacao_podio"]
    y_test = test["finish_position"].between(1, 3).astype(int).to_numpy()
    # Brier do baseline constante: media de (y_teste - p_treino)^2
    expected_brier = float(np.mean((y_test - train_prev) ** 2))
    assert abs(clf["constant_brier_score"] - expected_brier) < 1e-9
    # A identidade Brier = p(1-p) NAO vale quando as prevalencias diferem.
    assert abs(clf["constant_brier_score"] - train_prev * (1 - train_prev)) > 1e-6


def test_insufficient_training_classes_do_not_crash():
    tables = _tables_for_years(TRAIN_YEARS + TEST_YEARS)
    # Todo o treino vira não-pódio: classificador não cabe, mas pipeline não quebra.
    tables["results"].loc[tables["results"]["year"] <= 2014, "positionOrder"] = 5
    df = build_dataset(tables)
    stats = evaluate_models(df)
    clf = stats["classificacao_podio"]
    assert clf["roc_auc"] is None
    assert clf["brier_score"] is None
    assert clf["erro_classificador"] is not None
    reg = stats["regressao_linear"]
    assert reg["mae"] is not None


def test_temporal_validation_has_no_chronological_leakage():
    df = _df_for_years([2013, 2014, 2015])
    rows = temporal_validation_regression(df, train_end_year=2015)
    assert [r["ano_validacao"] for r in rows] == [2014, 2015]
    for row in rows:
        expected_train = int((df["year"] < row["ano_validacao"]).sum())
        assert row["amostras_treino"] == expected_train
        assert row["amostras_validacao"] == int((df["year"] == row["ano_validacao"]).sum())
        # Modelo nunca é pior que o baseline de grid em todos os anos (não é exigência,
        # apenas confirma que as duas séries foram computadas no mesmo recorte).
        assert row["mae"] is not None and row["mae_grid_baseline"] is not None


def test_temporal_classification_has_no_chronological_leakage():
    df = _df_for_years([2013, 2014, 2015])
    rows = temporal_validation_classification(df, train_end_year=2015)
    assert [r["ano_validacao"] for r in rows] == [2014, 2015]
    for row in rows:
        expected_train = int((df["year"] < row["ano_validacao"]).sum())
        assert row["amostras_treino"] == expected_train
        assert row["roc_auc"] is not None
        assert row["grid_ranking_auc"] is not None


def test_per_season_test_keys_match_test_years():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    stats = evaluate_models(df)
    test_years = {str(y) for y in TEST_YEARS}
    assert set(stats["regressao_linear"]["per_season_teste"].keys()) <= test_years
    assert set(stats["classificacao_podio"]["per_season_teste"].keys()) <= test_years
    for year in TEST_YEARS:
        row = stats["classificacao_podio"]["per_season_teste"][str(year)]
        cm = row["confusion_matrix"]
        assert sum(sum(r) for r in cm) == int((df["year"] == year).sum())


def test_metrics_are_json_serializable_without_nan():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    stats = evaluate_models(df)
    text = json.dumps(stats, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_classifier_metrics_consistency():
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    stats, arrays = evaluate_models_with_arrays(df)
    clf = stats["classificacao_podio"]
    p = arrays["classifier_probabilities_test"]
    y = arrays["target_test"]
    thr = clf["threshold"]
    y_pred = (p >= thr).astype(int)
    cm = clf["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    assert tp + tn + fp + fn == len(y)
    assert clf["true_positive"] == tp
    assert clf["false_positive"] == fp
    assert clf["true_negative"] == tn
    assert clf["false_negative"] == fn
    assert abs(clf["threshold"] - arrays["classifier_threshold"]) < 1e-9


def test_save_model_figures_creates_png(tmp_path):
    df = _df_for_years(TRAIN_YEARS + TEST_YEARS)
    stats, arrays = evaluate_models_with_arrays(df)
    out = tmp_path / "figures"
    saved = save_model_figures(stats, arrays, out)
    names = {p.name for p in saved}
    assert names == {
        "modelo_mae_comparacao.png",
        "modelo_observado_vs_previsto.png",
        "modelo_distribuicao_erros.png",
        "modelo_curva_roc.png",
        "modelo_curva_pr.png",
        "modelo_curva_calibracao.png",
        "modelo_validacao_temporal_treino.png",
    }
    for path in saved:
        assert path.exists()
        assert path.read_bytes()[:4] == b"\x89PNG"


def test_regression_diagnostics_on_constant_target_returns_none_r2():
    y_true = pd.Series([5.0, 5.0, 5.0])
    y_pred = pd.Series([5.0, 4.0, 6.0])
    diag = regression_diagnostics(y_true, y_pred)
    assert diag["r2"] is None
    assert diag["mae"] is not None


def test_run_analysis_end_to_end_on_synthetic_data(synthetic_tables, monkeypatch, tmp_path):
    """Execução completa da análise sobre dados sintéticos: métricas e figuras."""
    monkeypatch.setattr(run_analysis, "ROOT", tmp_path)
    monkeypatch.setattr(run_analysis, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    monkeypatch.setattr(run_analysis, "save_figures", lambda *args: None)
    run_analysis.main()

    metrics = json.loads((tmp_path / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    assert "modelagem" in metrics
    m = metrics["modelagem"]
    assert m["regressao_linear"]["rmse"] is not None
    assert m["regressao_linear"]["per_season_teste"]
    assert m["classificacao_podio"]["train_prevalence"] is not None
    assert m["classificacao_podio"]["erro_classificador"] is None
    assert m["classificacao_podio"]["threshold"] is not None
    # Nenhum NaN literal no JSON exportado.
    raw = (tmp_path / "outputs" / "metrics.json").read_bytes()
    assert b"NaN" not in raw and b"Infinity" not in raw

    figs = tmp_path / "outputs" / "figures"
    produced = {p.name for p in figs.iterdir() if p.suffix == ".png"}
    assert produced == EXPECTED_MODEL_PNGS
    for path in figs.iterdir():
        if path.suffix == ".png":
            assert path.read_bytes()[:4] == b"\x89PNG"


def test_classification_diagnostics_single_class_returns_none_auc():
    y_true = pd.Series([0, 0, 0])
    y_proba = pd.Series([0.1, 0.2, 0.3])
    grids = pd.Series([1, 2, 3])
    diag = classification_diagnostics(y_true, y_proba, grids, 0.5, 0.0)
    assert diag["roc_auc"] is None
    assert diag["pr_auc"] is None
    assert diag["constant_roc_auc"] is None


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/raw/results.csv").exists(),
    reason="CSVs históricos não disponíveis neste ambiente.",
)
def test_historical_reference_metrics_unchanged_on_local_data():
    from src.pipeline import load_data
    from pathlib import Path

    tables = load_data(Path("data/raw"))
    df = build_dataset(tables)
    stats = evaluate_models(df)
    reg = stats["regressao_linear"]
    refs = stats["referencias"]
    clf = stats["classificacao_podio"]
    assert reg["mae"] == 3.7748
    assert reg["r2"] == 0.383
    assert refs["mae_previsao_posicao_igual_ao_grid"] == 3.6926
    assert abs(clf["roc_auc"] - 0.9399) < 1e-4
    assert abs(clf["brier_score"] - 0.0665) < 1e-4
