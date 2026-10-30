"""Gráficos estáticos e reproduzíveis da avaliação de modelagem (Fase 2).

Todas as figuras são derivadas exclusivamente dos resultados computados por
``src.model.evaluate_models_with_arrays``. Não há números estáticos nem
afirmações causais embutidas nos gráficos.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paleta atribuída pela função da marca, nunca pela posição na série.
# Azul = modelo ajustado; vermelho = referência simples (grid); cinza = referência constante / aleatória.
MODEL = "#0072B2"
GRID_BASELINE = "#D55E00"
CONSTANT_BASELINE = "#999999"
DIAGONAL = "#555555"
INK = "#333333"
GRID_COLOR = "#e8e8e8"

DPI = 150


def _save(fig: plt.Figure, output: Path, name: str) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _style_axes(ax):
    ax.tick_params(colors=INK, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _title(ax, text: str, train_label: str, test_label: str):
    ax.set_title(
        f"{text}  |  Treino {train_label} · Teste {test_label}",
        fontsize=11,
        color=INK,
        pad=10,
    )


def save_mae_comparison(stats: dict, output: Path) -> list[Path]:
    """Barras de MAE no teste final: modelo versus referências."""
    reg = stats["regressao_linear"]
    refs = stats["referencias"]
    labels = ["Regressão linear", "Previsão = grid", "Mediana do treino"]
    values = [reg["mae"], refs["mae_previsao_posicao_igual_ao_grid"], refs["mae_previsao_constante_mediana"]]
    colors = [MODEL, GRID_BASELINE, CONSTANT_BASELINE]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        if value is not None:
            ax.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
                color=INK,
            )
    ax.set_ylabel("MAE (posições, quanto menor melhor)", color=INK)
    ax.set_ylim(0, max([v for v in values if v is not None] or [1]) * 1.25)
    _style_axes(ax)
    _title(ax, "Erro médio absoluto no teste final", stats["treino_anos"], stats["teste_anos"])
    fig.tight_layout()
    return [_save(fig, output, "modelo_mae_comparacao.png")]


def save_observed_vs_predicted(arrays: dict, output: Path) -> list[Path]:
    """Dispersão ordenação registrada observada versus prevista."""
    y_true = arrays["finish_position_test"]
    y_pred = arrays["regression_predictions_test"]
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.scatter(y_true, y_pred, s=16, color=MODEL, alpha=0.65, edgecolors="none")
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=DIAGONAL, linewidth=1, linestyle="--")
    ax.set_xlabel("Ordenação final registrada (observada)", color=INK)
    ax.set_ylabel("Ordenação final prevista pelo modelo", color=INK)
    _style_axes(ax)
    ax.set_title("Observado versus previsto (regressão linear, teste final)", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    return [_save(fig, output, "modelo_observado_vs_previsto.png")]


def save_error_distribution(arrays: dict, output: Path) -> list[Path]:
    """Histograma dos resíduos (observado menos previsto)."""
    errors = arrays["regression_errors"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(errors, bins=20, color=MODEL, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=DIAGONAL, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Resíduo = observado − previsto (posições)", color=INK)
    ax.set_ylabel("Quantidade de resultados", color=INK)
    _style_axes(ax)
    ax.set_title("Distribuição dos erros de previsão no teste final", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    return [_save(fig, output, "modelo_distribuicao_erros.png")]


def _roc_curve(y_true, score):
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order]
    score = score[order]
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    prev = None
    for i, s in enumerate(score):
        if s != prev:
            prev = s
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
    tpr.append(1.0)
    fpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def save_roc_curve(stats: dict, arrays: dict, output: Path) -> list[Path]:
    """Curva ROC do classificador de pódio versus ranking por grid."""
    p_test = arrays["classifier_probabilities_test"]
    y_test = arrays["target_test"]
    grid_test = arrays["grid_test"]
    clf = stats["classificacao_podio"]
    if p_test is None:
        return []
    fpr_m, tpr_m = _roc_curve(y_test, p_test)
    fpr_g, tpr_g = _roc_curve(y_test, -grid_test)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot(fpr_m, tpr_m, color=MODEL, linewidth=1.8, label="Regressão logística")
    ax.plot(fpr_g, tpr_g, color=GRID_BASELINE, linewidth=1.4, linestyle=":", label="Ranking por −grid")
    ax.plot([0, 1], [0, 1], color=DIAGONAL, linewidth=1, linestyle="--")
    ax.set_xlabel("Taxa de falsos positivos", color=INK)
    ax.set_ylabel("Taxa de verdadeiros positivos", color=INK)
    _style_axes(ax)
    ax.legend(fontsize=9, loc="lower right")
    auc_m = clf["roc_auc"]
    auc_g = clf["grid_ranking_auc"]
    label = f"AUC logística = {auc_m:.4f}" if auc_m is not None else "AUC logística = n/d"
    label2 = f"AUC −grid = {auc_g:.4f}" if auc_g is not None else "AUC −grid = n/d"
    ax.text(
        0.02,
        0.02,
        f"{label}\n{label2}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK,
        verticalalignment="bottom",
    )
    ax.set_title("Curva ROC — evento pódio no teste final", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    return [_save(fig, output, "modelo_curva_roc.png")]


def save_pr_curve(stats: dict, arrays: dict, output: Path) -> list[Path]:
    """Curva precisão-recall do classificador versus prevalência constante."""
    p_test = arrays["classifier_probabilities_test"]
    y_test = arrays["target_test"]
    clf = stats["classificacao_podio"]
    if p_test is None:
        return []
    order = np.argsort(-p_test, kind="mergesort")
    y = y_test[order]
    score = p_test[order]
    n_pos = float(np.sum(y == 1))
    if n_pos == 0:
        return []
    precision = [1.0]
    recall = [0.0]
    tp = 0
    fp = 0
    prev = None
    for i, s in enumerate(score):
        if s != prev:
            prev = s
            denom = tp + fp
            precision.append(tp / denom if denom else 0.0)
            recall.append(tp / n_pos)
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
    precision.append(tp / (tp + fp) if (tp + fp) else 0.0)
    recall.append(1.0)
    precision = np.array(precision)
    recall = np.array(recall)
    prevalence = arrays["train_prevalence"]

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot(recall, precision, color=MODEL, linewidth=1.8, label="Regressão logística")
    ax.axhline(prevalence, color=CONSTANT_BASELINE, linewidth=1.4, linestyle=":", label="Prevalência constante (treino)")
    ax.set_xlabel("Recall", color=INK)
    ax.set_ylabel("Precisão", color=INK)
    _style_axes(ax)
    ax.legend(fontsize=9, loc="lower left")
    pr = clf["pr_auc"]
    ax.text(
        0.02,
        0.98,
        f"AP = {pr:.4f}" if pr is not None else "AP = n/d",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK,
        verticalalignment="top",
    )
    ax.set_title("Curva precisão-recall — evento pódio no teste final", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    return [_save(fig, output, "modelo_curva_pr.png")]


def save_calibration_curve(stats: dict, output: Path) -> list[Path]:
    """Curva de calibração com bins de probabilidade prevista."""
    clf = stats["classificacao_podio"]
    bins = clf["calibration_bins"] or []
    if not bins:
        return []
    xs = [b["mean_predicted"] for b in bins]
    ys = [b["fraction_positives"] if b["fraction_positives"] is not None else np.nan for b in bins]
    counts = [b["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    sizes = [20 + 180 * (c / max(counts)) if max(counts) else 40 for c in counts]
    valid = [i for i in range(len(xs)) if not math.isnan(ys[i])]
    ax.plot([0, 1], [0, 1], color=DIAGONAL, linewidth=1, linestyle="--", label="Calibrado")
    if valid:
        ax.plot(
            [xs[i] for i in valid],
            [ys[i] for i in valid],
            color=MODEL,
            marker="o",
            linestyle="-",
            linewidth=1.6,
            markersize=5,
            label="Modelo",
        )
        ax.scatter(
            [xs[i] for i in valid],
            [ys[i] for i in valid],
            s=[sizes[i] for i in valid],
            color=MODEL,
            alpha=0.5,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("Probabilidade prevista (média do bin)", color=INK)
    ax.set_ylabel("Fração de pódios observada", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style_axes(ax)
    ax.legend(fontsize=9, loc="upper left")
    ece = clf["calibration_ece"]
    mce = clf["calibration_mce"]
    ax.text(
        0.98,
        0.02,
        f"ECE = {ece:.4f}\nMCE = {mce:.4f}" if ece is not None else "ECE/MCE = n/d",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK,
        verticalalignment="bottom",
        horizontalalignment="right",
    )
    ax.set_title("Calibração do classificador de pódio (teste final)", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    return [_save(fig, output, "modelo_curva_calibracao.png")]


def save_temporal_validation(stats: dict, output: Path) -> list[Path]:
    """Validação temporal progressiva dentro do treino: regressão e classificação."""
    reg_rows = stats["validacao_temporal_treino"]["regressao"]
    clf_rows = stats["validacao_temporal_treino"]["classificacao"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    ax = axes[0]
    if reg_rows:
        years = [r["ano_validacao"] for r in reg_rows]
        mae = [r["mae"] for r in reg_rows]
        mae_g = [r["mae_grid_baseline"] for r in reg_rows]
        ax.plot(years, mae, color=MODEL, marker="o", linewidth=1.6, markersize=4, label="Regressão linear")
        ax.plot(years, mae_g, color=GRID_BASELINE, marker="o", linewidth=1.4, linestyle=":", markersize=4, label="Previsão = grid")
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Ano de validação", color=INK)
    ax.set_ylabel("MAE (posições)", color=INK)
    _style_axes(ax)
    ax.set_title("Regressão — validação temporal no treino", fontsize=10, color=INK, pad=8)

    ax = axes[1]
    if clf_rows:
        years = [r["ano_validacao"] for r in clf_rows]
        auc = [r["roc_auc"] for r in clf_rows]
        auc_g = [r["grid_ranking_auc"] for r in clf_rows]
        ax.plot(years, auc, color=MODEL, marker="o", linewidth=1.6, markersize=4, label="Regressão logística")
        ax.plot(years, auc_g, color=GRID_BASELINE, marker="o", linewidth=1.4, linestyle=":", markersize=4, label="Ranking por −grid")
        ax.legend(fontsize=8, loc="best")
    ax.set_xlabel("Ano de validação", color=INK)
    ax.set_ylabel("ROC AUC", color=INK)
    _style_axes(ax)
    ax.set_title("Classificação — validação temporal no treino", fontsize=10, color=INK, pad=8)

    fig.suptitle(
        f"Validação temporal progressiva — treino {stats['treino_anos']} (apenas anos anteriores)",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return [_save(fig, output, "modelo_validacao_temporal_treino.png")]


def save_model_figures(stats: dict, arrays: dict, output: Path) -> list[Path]:
    """Gera todas as figuras de modelagem e retorna os caminhos salvos."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved += save_mae_comparison(stats, output)
    saved += save_observed_vs_predicted(arrays, output)
    saved += save_error_distribution(arrays, output)
    saved += save_roc_curve(stats, arrays, output)
    saved += save_pr_curve(stats, arrays, output)
    saved += save_calibration_curve(stats, output)
    saved += save_temporal_validation(stats, output)
    return saved
