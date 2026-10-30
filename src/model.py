"""Modelos com separação temporal para evitar misturar corridas do treino e teste.

Estende a avaliação original preservando as chaves históricas (
``mae``, ``r2``, ``roc_auc``, ``brier_score``, ``treino_anos``,
``teste_anos``, ``amostras_treino``, ``amostras_teste``,
``referencias.mae_previsao_posicao_igual_ao_grid``) e adiciona:
RMSE, distribuição de erros, predições fora do intervalo válido,
métricas por temporada, validação temporal progressiva apenas dentro do treino
(2004–2014), baseline de probabilidade constante calculada exclusivamente no
treino, PR-AUC, log-loss, calibração (ECE/MCE), matriz de confusão no limiar
escolhido no treino e AUC de ranking baseado em ``-grid``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


TRAIN_END_YEAR = 2014
TEST_START_YEAR = 2015


def to_json_safe(value: Any) -> Any:
    """Converte o resultado para tipos serializáveis em JSON padrão.

    Valores ausentes/infinitos viram None; não substitui desconhecimento por
    zero nem altera denominadores.
    """
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    if hasattr(value, "item"):
        try:
            return to_json_safe(value.item())
        except Exception:
            return str(value)
    return value


def _as_float_array(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def regression_diagnostics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    grids: pd.Series | np.ndarray | None = None,
) -> dict[str, Any]:
    """Métricas de regressão para posição final registrada.

    A posição é ordinal; a regressão contínua pode prever fora do intervalo
    válido e não garante ordenação coerente entre pilotos.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if grids is not None:
        grids = _as_float_array(grids)
    else:
        grids = y_true.copy()

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(grids)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    grids = grids[mask]
    n = int(len(y_true))

    empty = {
        "n": 0,
        "mae": None,
        "rmse": None,
        "r2": None,
        "mean_error": None,
        "median_error": None,
        "mean_abs_error": None,
        "std_error": None,
        "outside_valid_range_count": 0,
        "outside_valid_range_pct": None,
        "valid_range_min": None,
        "valid_range_max": None,
    }
    if n == 0:
        return empty

    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if n > 1 and np.unique(y_true).size > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = None

    valid_min = 1.0
    valid_max = float(max(np.max(y_true), np.max(grids))) if n else float(np.max(y_true))
    outside = (y_pred < valid_min) | (y_pred > valid_max)
    outside_count = int(np.sum(outside))
    outside_pct = float(100.0 * outside_count / n)

    return {
        "n": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
        "mean_error": round(float(np.mean(errors)), 4),
        "median_error": round(float(np.median(errors)), 4),
        "mean_abs_error": round(float(np.mean(abs_errors)), 4),
        "std_error": round(float(np.std(errors, ddof=1)), 4) if n > 1 else None,
        "outside_valid_range_count": outside_count,
        "outside_valid_range_pct": round(outside_pct, 2),
        "valid_range_min": round(valid_min, 2),
        "valid_range_max": round(valid_max, 2),
    }


def regression_per_season(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, dict[str, Any]]:
    """Métricas de regressão por temporada do conjunto de teste."""
    result: dict[str, dict[str, Any]] = {}
    if df.empty or len(y_pred) != len(df):
        return result
    tmp = df.copy()
    tmp["_y_true"] = tmp["finish_position"].astype(float).to_numpy()
    tmp["_y_pred"] = np.asarray(y_pred, dtype=float)
    tmp["_grid"] = tmp["grid"].astype(float).to_numpy()
    for year, group in tmp.groupby("year", sort=True):
        diag = regression_diagnostics(
            group["_y_true"], group["_y_pred"], group["_grid"]
        )
        result[str(int(year))] = {**diag, "modelo": model_name}
    return result


def classification_diagnostics(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    grids: pd.Series | np.ndarray,
    threshold: float,
    train_prevalence: float,
) -> dict[str, Any]:
    """Métricas de classificação de pódio com limiar definido apenas no treino."""
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    grids = np.asarray(grids, dtype=float)
    mask = np.isfinite(y_proba) & np.isfinite(grids)
    y_true = y_true[mask]
    y_proba = y_proba[mask]
    grids = grids[mask]
    n = int(len(y_true))

    y_pred = (y_proba >= float(threshold)).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    both_classes = np.unique(y_true).size > 1
    try:
        roc_auc = float(roc_auc_score(y_true, y_proba)) if both_classes else None
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, y_proba)) if both_classes else None
    except ValueError:
        pr_auc = None
    try:
        logloss = float(log_loss(y_true, y_proba, labels=[0, 1])) if both_classes else None
    except ValueError:
        logloss = None
    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except ValueError:
        brier = None

    try:
        grid_ranking_auc = float(roc_auc_score(y_true, -grids)) if both_classes else None
    except ValueError:
        grid_ranking_auc = None

    bin_edges = np.linspace(0.0, 1.0, 11)
    calibration_bins = []
    ece_numerator = 0.0
    mce = 0.0
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < 9:
            bin_mask = (y_proba >= lo) & (y_proba < hi)
        else:
            bin_mask = (y_proba >= lo) & (y_proba <= hi)
        count = int(np.sum(bin_mask))
        if count == 0:
            calibration_bins.append(
                {
                    "bin": i + 1,
                    "mean_predicted": round(float((lo + hi) / 2), 4),
                    "fraction_positives": None,
                    "count": 0,
                }
            )
            continue
        mean_pred = float(np.mean(y_proba[bin_mask]))
        frac_pos = float(np.mean(y_true[bin_mask]))
        calibration_bins.append(
            {
                "bin": i + 1,
                "mean_predicted": round(mean_pred, 4),
                "fraction_positives": round(frac_pos, 4),
                "count": count,
            }
        )
        weight = count / n
        ece_numerator += weight * abs(frac_pos - mean_pred)
        mce = max(mce, abs(frac_pos - mean_pred))

    p_const = float(np.clip(train_prevalence, 1e-12, 1 - 1e-12))
    # Baseline de probabilidade constante: a probabilidade é estimada exclusivamente
    # no treino (train_prevalence); os rótulos usados para pontuar são os do conjunto
    # sendo avaliado (y_true), nunca para ajustar o modelo. Para predições constantes
    # o Average Precision equivale à fração de positivos do conjunto pontuado.
    try:
        constant_pr_auc = (
            float(
                average_precision_score(
                    y_true, np.full(y_true.shape, p_const, dtype=float)
                )
            )
            if both_classes
            else None
        )
    except ValueError:
        # Para predições constantes, o Average Precision é a fração de positivos
        # do conjunto pontuado; usado como fallback robusto.
        constant_pr_auc = float(np.mean(y_true)) if both_classes else None
    try:
        constant_roc_auc = 0.5 if both_classes else None
    except ValueError:
        constant_roc_auc = None
    try:
        constant_logloss = float(
            -np.mean(
                y_true * np.log(p_const) + (1 - y_true) * np.log(1 - p_const)
            )
        )
    except ValueError:
        constant_logloss = None
    try:
        constant_brier = float(np.mean((y_true - p_const) ** 2))
    except ValueError:
        constant_brier = None

    return {
        "n": n,
        "threshold": round(float(threshold), 4),
        "threshold_criterion": "youden_no_treino",
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": {0: int(np.sum(y_true == 0)), 1: int(np.sum(y_true == 1))},
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "log_loss": round(logloss, 4) if logloss is not None else None,
        "brier_score": round(brier, 4) if brier is not None else None,
        "grid_ranking_auc": round(grid_ranking_auc, 4) if grid_ranking_auc is not None else None,
        # Precisão numérica preservada (sem arredondamento de apresentação): usada em
        # contratos de teste com tolerância estrita e como probabilidade de referência.
        "train_prevalence": float(train_prevalence),
        "constant_roc_auc": round(constant_roc_auc, 4) if constant_roc_auc is not None else None,
        "constant_pr_auc": constant_pr_auc,
        "constant_log_loss": round(constant_logloss, 4) if constant_logloss is not None else None,
        # Precisão numérica preservada: Brier do baseline constante calculado sobre os
        # rótulos do conjunto avaliado com a probabilidade p estimada apenas no treino.
        "constant_brier_score": constant_brier,
        "calibration_ece": round(ece_numerator, 4),
        "calibration_mce": round(mce, 4),
        "calibration_bins": calibration_bins,
    }


def classification_per_season(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float,
    train_prevalence: float,
) -> dict[str, dict[str, Any]]:
    """Métricas de classificação por temporada do conjunto de teste."""
    result: dict[str, dict[str, Any]] = {}
    if df.empty or len(y_proba) != len(df):
        return result
    tmp = df.copy()
    tmp["_y_true"] = tmp["finish_position"].between(1, 3).astype(int).to_numpy()
    tmp["_y_proba"] = np.asarray(y_proba, dtype=float)
    tmp["_grid"] = tmp["grid"].astype(float).to_numpy()
    for year, group in tmp.groupby("year", sort=True):
        result[str(int(year))] = classification_diagnostics(
            group["_y_true"],
            group["_y_proba"],
            group["_grid"],
            threshold,
            train_prevalence,
        )
    return result


def select_threshold_train(y_train: pd.Series | np.ndarray, y_proba_train: pd.Series | np.ndarray) -> float:
    """Escolhe limiar apenas com dados de treino, maximizando Youden.

    Nunca consulta o conjunto de teste final.
    """
    y_train = np.asarray(y_train, dtype=int)
    y_proba_train = np.asarray(y_proba_train, dtype=float)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_youden = -np.inf
    for thr in thresholds:
        y_pred = (y_proba_train >= thr).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
        except ValueError:
            tn = fp = fn = tp = 0
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        youden = sensitivity + specificity - 1.0
        tie = abs(youden - best_youden) <= 1e-12
        if youden > best_youden + 1e-12 or (
            tie and abs(float(thr) - 0.5) < abs(best_threshold - 0.5)
        ):
            best_youden = youden
            best_threshold = float(thr)
    return best_threshold


def temporal_validation_regression(
    df: pd.DataFrame, train_end_year: int = TRAIN_END_YEAR
) -> list[dict[str, Any]]:
    """Validação temporal progressiva dentro do período de treino.

    Cada ano de validação usa apenas temporadas anteriores como treino,
    sem vazamento cronológico.
    """
    train = df.loc[df["year"] <= train_end_year].copy()
    years = sorted(train["year"].unique())
    rows: list[dict[str, Any]] = []
    for i, val_year in enumerate(years):
        if i == 0:
            continue
        train_win = train.loc[train["year"] < val_year]
        val_win = train.loc[train["year"] == val_year]
        if train_win.empty or val_win.empty:
            continue
        try:
            reg = LinearRegression().fit(train_win[["grid"]], train_win["finish_position"])
            pred = reg.predict(val_win[["grid"]])
        except Exception:
            continue
        diag = regression_diagnostics(
            val_win["finish_position"], pred, val_win["grid"]
        )
        baseline = regression_diagnostics(
            val_win["finish_position"], val_win["grid"].astype(float), val_win["grid"]
        )
        rows.append(
            {
                "ano_validacao": int(val_year),
                "amostras_treino": int(len(train_win)),
                "amostras_validacao": int(len(val_win)),
                "mae": diag["mae"],
                "rmse": diag["rmse"],
                "r2": diag["r2"],
                "mae_grid_baseline": baseline["mae"],
                "rmse_grid_baseline": baseline["rmse"],
                "r2_grid_baseline": baseline["r2"],
            }
        )
    return rows


def temporal_validation_classification(
    df: pd.DataFrame, train_end_year: int = TRAIN_END_YEAR
) -> list[dict[str, Any]]:
    """Validação temporal progressiva do classificador dentro do treino."""
    train = df.loc[df["year"] <= train_end_year].copy()
    years = sorted(train["year"].unique())
    rows: list[dict[str, Any]] = []
    for i, val_year in enumerate(years):
        if i == 0:
            continue
        train_win = train.loc[train["year"] < val_year]
        val_win = train.loc[train["year"] == val_year]
        if train_win.empty or val_win.empty:
            continue
        y_train = train_win["finish_position"].between(1, 3).astype(int).to_numpy()
        y_val = val_win["finish_position"].between(1, 3).astype(int).to_numpy()
        try:
            clf = LogisticRegression(max_iter=1000, random_state=42).fit(
                train_win[["grid"]], y_train
            )
        except Exception:
            continue
        p_train = clf.predict_proba(train_win[["grid"]])[:, 1]
        p_val = clf.predict_proba(val_win[["grid"]])[:, 1]
        threshold = select_threshold_train(y_train, p_train)
        try:
            diag = classification_diagnostics(
                y_val, p_val, val_win["grid"].astype(float), threshold, float(np.mean(y_train))
            )
        except Exception:
            continue
        rows.append(
            {
                "ano_validacao": int(val_year),
                "amostras_treino": int(len(train_win)),
                "amostras_validacao": int(len(val_win)),
                "prevalencia_treino": round(float(np.mean(y_train)), 4),
                "threshold": diag["threshold"],
                "roc_auc": diag["roc_auc"],
                "brier_score": diag["brier_score"],
                "precision": diag["precision"],
                "recall": diag["recall"],
                "f1": diag["f1"],
                "grid_ranking_auc": diag["grid_ranking_auc"],
            }
        )
    return rows


def _fit_regression(train: pd.DataFrame):
    reg = LinearRegression().fit(train[["grid"]], train["finish_position"])
    return reg


def _fit_classifier(train: pd.DataFrame):
    y_train = train["finish_position"].between(1, 3).astype(int).to_numpy()
    if np.unique(y_train).size < 2:
        raise ValueError("Classes insuficientes no treino para classificação.")
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(train[["grid"]], y_train)
    return clf, y_train


def evaluate_models(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Avalia modelos com separação temporal e devolve métricas serializáveis.

    Os valores históricos publicados (MAE linear, baseline grid, AUC e Brier
    da regressão logística) são mantidos; novas métricas são adicionadas sem
    remover as existentes.
    """
    train = df.loc[df["year"] <= TRAIN_END_YEAR].copy()
    test = df.loc[df["year"] >= TEST_START_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError(
            "É necessário ter dados de treinamento até 2014 e de teste a partir de 2015."
        )

    x_train = train[["grid"]]
    y_train = train["finish_position"]
    x_test = test[["grid"]]
    y_test = test["finish_position"]

    reg = _fit_regression(train)
    predictions = reg.predict(x_test)
    reg_diag = regression_diagnostics(y_test, predictions, x_test["grid"])
    grid_diag = regression_diagnostics(
        y_test, x_test["grid"].astype(float), x_test["grid"]
    )
    median_pred = np.full_like(y_test.to_numpy(dtype=float), float(np.median(y_train)))
    median_diag = regression_diagnostics(y_test, median_pred, x_test["grid"])

    target_train = train["finish_position"].between(1, 3).astype(int)
    target_test = test["finish_position"].between(1, 3).astype(int)
    train_prevalence = float(np.mean(target_train))

    try:
        clf, _ = _fit_classifier(train)
        p_test = clf.predict_proba(x_test)[:, 1]
        p_train = clf.predict_proba(x_train)[:, 1]
        threshold = select_threshold_train(target_train, p_train)
        clf_diag = classification_diagnostics(
            target_test.to_numpy(), p_test, x_test["grid"].astype(float), threshold, train_prevalence
        )
        classifier_error = None
    except Exception as exc:  # mantém execução mesmo se o classificador não couber
        p_test = None
        p_train = None
        threshold = None
        clf_diag = None
        classifier_error = str(exc)

    train_years = sorted(train["year"].unique())
    test_years = sorted(test["year"].unique())

    def year_label(years_list):
        if not years_list:
            return None
        if len(years_list) == 1:
            return str(int(years_list[0]))
        return f"{int(years_list[0])}–{int(years_list[-1])}"

    stats: dict[str, Any] = {
        "treino_anos": year_label(train_years),
        "teste_anos": year_label(test_years),
        "amostras_treino": int(len(train)),
        "amostras_teste": int(len(test)),
        "regressao_linear": {
            "mae": reg_diag["mae"],
            "rmse": reg_diag["rmse"],
            "r2": reg_diag["r2"],
            "coeficiente_grid": round(float(reg.coef_[0]), 4),
            "intercepto": round(float(reg.intercept_), 4),
            "mean_error": reg_diag["mean_error"],
            "median_error": reg_diag["median_error"],
            "mean_abs_error": reg_diag["mean_abs_error"],
            "std_error": reg_diag["std_error"],
            "outside_valid_range_count": reg_diag["outside_valid_range_count"],
            "outside_valid_range_pct": reg_diag["outside_valid_range_pct"],
            "valid_range_min": reg_diag["valid_range_min"],
            "valid_range_max": reg_diag["valid_range_max"],
            "per_season_teste": regression_per_season(test, predictions, "regressao_linear"),
        },
        "referencias": {
            "mae_previsao_posicao_igual_ao_grid": grid_diag["mae"],
            "rmse_previsao_posicao_igual_ao_grid": grid_diag["rmse"],
            "r2_previsao_posicao_igual_ao_grid": grid_diag["r2"],
            "mae_previsao_constante_mediana": median_diag["mae"],
            "rmse_previsao_constante_mediana": median_diag["rmse"],
            "r2_previsao_constante_mediana": median_diag["r2"],
        },
        "classificacao_podio": {
            "threshold": clf_diag["threshold"] if clf_diag else None,
            "threshold_criterion": clf_diag["threshold_criterion"] if clf_diag else None,
            "roc_auc": clf_diag["roc_auc"] if clf_diag else None,
            "pr_auc": clf_diag["pr_auc"] if clf_diag else None,
            "log_loss": clf_diag["log_loss"] if clf_diag else None,
            "brier_score": clf_diag["brier_score"] if clf_diag else None,
            "confusion_matrix": clf_diag["confusion_matrix"] if clf_diag else None,
            "precision": clf_diag["precision"] if clf_diag else None,
            "recall": clf_diag["recall"] if clf_diag else None,
            "f1": clf_diag["f1"] if clf_diag else None,
            "support": clf_diag["support"] if clf_diag else None,
            "grid_ranking_auc": clf_diag["grid_ranking_auc"] if clf_diag else None,
            # Contagens da matriz de confusão no limiar escolhido apenas no treino.
            "true_positive": clf_diag["true_positive"] if clf_diag else None,
            "true_negative": clf_diag["true_negative"] if clf_diag else None,
            "false_positive": clf_diag["false_positive"] if clf_diag else None,
            "false_negative": clf_diag["false_negative"] if clf_diag else None,
            # Precisão numérica preservada (sem arredondamento de apresentação).
            "train_prevalence": train_prevalence,
            "constant_roc_auc": clf_diag["constant_roc_auc"] if clf_diag else None,
            "constant_pr_auc": clf_diag["constant_pr_auc"] if clf_diag else None,
            "constant_log_loss": clf_diag["constant_log_loss"] if clf_diag else None,
            "constant_brier_score": clf_diag["constant_brier_score"] if clf_diag else None,
            "calibration_ece": clf_diag["calibration_ece"] if clf_diag else None,
            "calibration_mce": clf_diag["calibration_mce"] if clf_diag else None,
            "calibration_bins": clf_diag["calibration_bins"] if clf_diag else [],
            "erro_classificador": classifier_error,
            "per_season_teste": classification_per_season(test, p_test, threshold, train_prevalence)
            if p_test is not None
            else {},
        },
        "validacao_temporal_treino": {
            "regressao": temporal_validation_regression(df, TRAIN_END_YEAR),
            "classificacao": temporal_validation_classification(df, TRAIN_END_YEAR),
        },
    }
    return to_json_safe(stats)


def evaluate_models_with_arrays(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    """Como evaluate_models, mas devolve arrays para geração de figuras.

    Os arrays não entram no JSON de métricas.
    """
    train = df.loc[df["year"] <= TRAIN_END_YEAR].copy()
    test = df.loc[df["year"] >= TEST_START_YEAR].copy()
    if train.empty or test.empty:
        raise ValueError(
            "É necessário ter dados de treinamento até 2014 e de teste a partir de 2015."
        )

    x_train = train[["grid"]]
    y_train = train["finish_position"]
    x_test = test[["grid"]]
    y_test = test["finish_position"]

    reg = _fit_regression(train)
    predictions = reg.predict(x_test)

    target_train = train["finish_position"].between(1, 3).astype(int).to_numpy()
    target_test = test["finish_position"].between(1, 3).astype(int).to_numpy()
    train_prevalence = float(np.mean(target_train))

    try:
        clf, _ = _fit_classifier(train)
        p_test = clf.predict_proba(x_test)[:, 1]
        p_train = clf.predict_proba(x_train)[:, 1]
        threshold = select_threshold_train(target_train, p_train)
        classifier_error = None
    except Exception as exc:
        clf = None
        p_test = None
        p_train = None
        threshold = None
        classifier_error = str(exc)

    stats = evaluate_models(df)
    arrays = {
        "y_test": y_test.to_numpy(dtype=float),
        "finish_position_test": y_test.to_numpy(),
        "grid_test": x_test["grid"].to_numpy(dtype=float),
        "regression_predictions_test": predictions,
        "regression_errors": y_test.to_numpy(dtype=float) - predictions,
        "y_train": y_train.to_numpy(dtype=float),
        "target_train": target_train,
        "target_test": target_test,
        "classifier_probabilities_test": p_test,
        "classifier_probabilities_train": p_train,
        "classifier_threshold": threshold,
        "classifier_error": classifier_error,
        "train_prevalence": train_prevalence,
    }
    return stats, arrays
