"""Testes unitarios das figuras Plotly (src/dashboard/charts.py).

Verificam que as series, valores, rotulos e unidades das figuras sao derivados
dos DataFrames/arrays recebidos, sem valores estaticos ou inventados.
"""

import math

import pandas as pd
import pytest

from src.dashboard.charts import (
    fig_calibration,
    fig_delta_historical,
    fig_error_distribution,
    fig_grid_probabilities,
    fig_heatmap,
    fig_lap_time_distribution,
    fig_mae_comparison,
    fig_observed_vs_predicted,
    fig_precision_recall,
    fig_roc,
    fig_temporal,
    fig_climate_means,
    fig_climate_quality,
    fig_climate_timeline,
    fig_historical_sensitivity,
)
from src.dashboard.theme import ACCENT_RED


def _proba_df() -> pd.DataFrame:
    return pd.DataFrame({
        "grid": [1, 2, 3],
        "corridas": [263, 265, 260],
        "vitoria_pct": [53.23, 21.50, 5.00],
        "podio_pct": [79.85, 50.00, 15.00],
        "top10_pct": [88.97, 70.00, 40.00],
        "pontos_pct": [80.00, 60.00, 25.00],
    })


def test_fig_grid_probabilities_valores_do_dataframe():
    fig = fig_grid_probabilities(_proba_df())
    assert len(fig.data) == 3
    nomes = [t.name for t in fig.data]
    assert nomes == ["Vitória", "Pódio", "Top 10"]
    # x vem da coluna grid do DataFrame
    assert list(fig.data[0].x) == [1, 2, 3]
    # y da serie vitoria vem do DataFrame
    assert list(fig.data[0].y) == [53.23, 21.50, 5.00]


def test_fig_grid_probabilities_vazio():
    fig = fig_grid_probabilities(pd.DataFrame(columns=["grid", "vitoria_pct", "podio_pct", "top10_pct"]))
    assert "Sem registros" in fig.layout.title.text


def test_fig_heatmap_dominio_dos_dados():
    crosstab = pd.crosstab(pd.Series([1, 1, 2]), pd.Series([1, 2, 1]))
    fig = fig_heatmap(crosstab)
    assert len(fig.data) == 1 and fig.data[0].type == "heatmap"
    # celulas somam ao total de registros
    assert int(fig.data[0].z.sum()) == 3
    assert list(fig.data[0].x) == ["1", "2"]
    assert list(fig.data[0].y) == ["1", "2"]


def test_fig_heatmap_vazio():
    fig = fig_heatmap(pd.DataFrame())
    assert "Sem registros" in fig.layout.title.text


def test_fig_heatmap_modo_contagem_absoluta():
    crosstab = pd.DataFrame([[2, 1], [0, 3]], index=[1, 2], columns=[1, 2])
    fig = fig_heatmap(crosstab, mode="count")
    assert [list(row) for row in fig.data[0].z] == [[2, 1], [0, 3]]
    assert fig.data[0].colorbar.title.text == "Registros"
    # o hover informa a celula e o denominador do grupo (total da linha)
    assert list(fig.data[0].customdata[0][1][:2]) == [1, 3]
    assert list(fig.data[0].customdata[1][0][:2]) == [0, 3]
    assert fig.data[0].customdata[0][1][2] == pytest.approx(100 / 3)
    assert "Denominador" in fig.data[0].hovertemplate
    assert "%{customdata[2]:.2f}%" in fig.data[0].hovertemplate


def test_fig_heatmap_modo_percentual_por_grid():
    """Percentual condicionado ao grid: denominador e a linha, nunca o total geral."""
    crosstab = pd.DataFrame([[2, 2], [1, 3]], index=[1, 2], columns=[1, 2])
    fig = fig_heatmap(crosstab, mode="percent")
    assert [list(row) for row in fig.data[0].z] == [[50.0, 50.0], [25.0, 75.0]]
    assert fig.data[0].colorbar.title.text == "% do grid"
    # cada linha soma 100%, nao 100% do total geral
    assert [round(float(sum(row)), 9) for row in fig.data[0].z] == [100.0, 100.0]
    assert list(fig.data[0].customdata[1][1]) == [3, 4, 75.0]
    assert "Denominador" in fig.data[0].hovertemplate
    assert "%{customdata[2]:.2f}%" in fig.data[0].hovertemplate
    assert fig.data[0].zmin == 0
    assert fig.data[0].zmax == 100


def test_fig_heatmap_percentual_com_linha_sem_registros():
    """Linha sem registros nao vira zero: permanece indefinida (NaN)."""
    crosstab = pd.DataFrame([[0, 0], [1, 1]], index=[1, 2], columns=[1, 2])
    fig = fig_heatmap(crosstab, mode="percent")
    assert math.isnan(fig.data[0].z[0][0])
    assert list(fig.data[0].z[1]) == [50.0, 50.0]


def test_fig_observed_vs_predicted_valores_dos_arrays():
    fig = fig_observed_vs_predicted([1, 2, 3], [1.1, 1.9, 3.2])
    assert len(fig.data) == 2  # pontos + diagonal de identidade
    assert fig.data[0].name == "Resultado"
    assert fig.data[1].name == "Identidade (observado = previsto)"


def test_fig_observed_vs_predicted_vazio():
    fig = fig_observed_vs_predicted([], [])
    assert "Sem previsões para exibir" in fig.layout.title.text


def test_fig_error_distribution_valores_dos_arrays():
    fig = fig_error_distribution([0.5, -0.2, 0.1])
    assert len(fig.data) == 1 and fig.data[0].type == "histogram"
    assert list(fig.data[0].x) == [0.5, -0.2, 0.1]


def test_fig_mae_comparacion_valores_do_dicionario():
    fig = fig_mae_comparison({
        "Modelo: regressao linear": 3.7748,
        "Referencia: previsao = grid": 3.6926,
    })
    assert len(fig.data) == 1 and fig.data[0].type == "bar"
    assert list(fig.data[0].y) == [3.7748, 3.6926]
    assert list(fig.data[0].x) == ["Modelo: regressao linear", "Referencia: previsao = grid"]


def test_fig_roc_e_precision_recall_usam_arrays():
    fpr = [0.0, 0.2, 1.0]
    tpr = [0.0, 0.85, 1.0]
    fig_roc_result = fig_roc(fpr, tpr, 0.9399)
    assert len(fig_roc_result.data) == 2
    assert list(fig_roc_result.data[0].x) == fpr
    assert list(fig_roc_result.data[0].y) == tpr
    assert "ROC AUC = 0.940" in fig_roc_result.layout.title.text

    recall = [0.0, 0.6, 1.0]
    precision = [1.0, 0.7, 0.4]
    fig_pr = fig_precision_recall(recall, precision, 0.75, 0.14)
    assert len(fig_pr.data) == 2
    assert list(fig_pr.data[0].x) == recall
    assert list(fig_pr.data[0].y) == precision
    assert "Average Precision = 0.750" in fig_pr.layout.title.text


def test_fig_roc_sem_auc():
    fig = fig_roc([0.0, 1.0], [0.0, 1.0], None)
    assert "ROC AUC =" not in fig.layout.title.text


def test_fig_calibration_bins():
    bins = [
        {"mean_predicted": 0.1, "fraction_positives": 0.05, "count": 10},
        {"mean_predicted": 0.5, "fraction_positives": 0.5, "count": 20},
    ]
    fig = fig_calibration(bins)
    assert len(fig.data) == 2  # calibracao + referencia perfeita
    assert list(fig.data[0].x) == [0.1, 0.5]


def test_fig_calibration_vazio():
    fig = fig_calibration([])
    assert "Sem bins" in fig.layout.title.text


def test_fig_temporal_linhas_do_dataframe():
    rows = [{"ano_validacao": 2015, "mae": 3.5}, {"ano_validacao": 2016, "mae": 3.2}]
    fig = fig_temporal(rows, "mae", "MAE por janela")
    assert list(fig.data[0].x) == ["2015", "2016"]
    assert list(fig.data[0].y) == [3.5, 3.2]


def test_fig_delta_historical_barras_horizontais():
    rows = [
        {"grand_prix": "GP A", "diferenca_2014_menos_2013_seg": 2.1},
        {"grand_prix": "GP B", "diferenca_2014_menos_2013_seg": -0.5},
    ]
    fig = fig_delta_historical(rows, "diferenca_2014_menos_2013_seg", "grand_prix", "Mediana por GP")
    assert fig.data[0].orientation == "h"
    assert list(fig.data[0].y) == ["GP A", "GP B"]
    assert list(fig.data[0].x) == [2.1, -0.5]


def test_fig_lap_time_distribution_titulo_e_eixos():
    diffs = [2.0, -2.0, 3.0, 1.5, -0.5]
    fig = fig_lap_time_distribution(diffs)
    assert fig.layout.title.text == "Distribuição das diferenças de tempo de volta"
    assert fig.layout.xaxis.title.text == "Diferença de tempo (segundos)"
    assert fig.layout.yaxis.title.text == "Quantidade de pares"
    assert fig.data[0].type == "histogram"
    assert "Quantidade de pares" in fig.data[0].hovertemplate


def test_fig_lap_time_distribution_sem_descartar_valores():
    """Cada par contribui com um valor; extremos sao mantidos."""
    diffs = [-50.0, 0.0, 100.0, 5.5, 5.5]
    fig = fig_lap_time_distribution(diffs)
    assert len(fig.data[0].x) == len(diffs)  # nenhum par descartado
    assert min(fig.data[0].x) == min(diffs)  # extremo negativo preservado
    assert max(fig.data[0].x) == max(diffs)  # extremo positivo preservado


def test_fig_lap_time_distribution_linha_referencia_zero():
    fig = fig_lap_time_distribution([1.0, -1.0, 2.0])
    linhas_zero = [
        s for s in fig.layout.shapes
        if getattr(s, "xref", None) == "x" and float(s.x0) == 0 and float(s.x1) == 0
    ]
    assert len(linhas_zero) == 1
    assert linhas_zero[0].line.color == ACCENT_RED
    assert linhas_zero[0].line.dash == "dash"


def test_fig_lap_time_distribution_ignora_valores_ausentes():
    fig = fig_lap_time_distribution([1.0, float("nan"), 2.0, None, 3.0])
    assert sorted(fig.data[0].x) == [1.0, 2.0, 3.0]


def test_fig_lap_time_distribution_vazio():
    fig = fig_lap_time_distribution([])
    assert "Sem diferenças de tempo de volta para exibir" in fig.layout.title.text
    assert len(fig.data) == 0


def test_fig_climate_timeline_usa_marcas_sem_linhas_e_rotulo_agendado():
    frame = pd.DataFrame({
        "season": ["2013"], "hours_from_scheduled": [-2.0], "measurement_c": [22.5],
        "observation_time_utc": ["2013-10-13T04:00:00Z"], "report_type": ["FM-12"],
        "quality_class": ["passed_all_qc"], "quality_class_label": ["Qualidade aprovada"],
        "quality_flag": ["1"], "station_name": ["Station"],
    })
    fig = fig_climate_timeline(frame, value_label="Temperatura do ar")
    assert fig.data[0].mode == "markers"
    assert "não é a largada real" in fig.layout.xaxis.title.text
    assert "UTC" in fig.data[0].hovertemplate
    assert fig.data[0].customdata[0][2] == "Qualidade aprovada"
    assert fig.data[0].customdata[0][3] == "passed_all_qc"


def test_fig_climate_means_and_quality_derive_values_from_rows():
    means = pd.DataFrame({
        "season": ["2013", "2014"], "grand_prix": ["GP", "GP"], "mean_c": [20.0, 25.0],
        "n": [4, 5], "station_name": ["Station", "Station"],
        "distance_to_circuit_km": [12.0, 12.0], "coverage": ["4 timestamps", "5 timestamps"],
    })
    fig = fig_climate_means(means, label="Média")
    assert list(fig.data[0].y) == [20.0]
    assert list(fig.data[1].y) == [25.0]
    quality = pd.DataFrame({
        "variable": ["TMP"], "variable_label": ["Temperatura do ar"],
        "quality_class": ["passed_all_qc"], "quality_class_label": ["Qualidade aprovada"],
        "quality_description": ["Passed"], "eligible": [True], "count": [4],
    })
    quality_fig = fig_climate_quality(quality)
    assert quality_fig.data[0].x[0] == 4
    assert "Temperatura do ar · Qualidade aprovada" in quality_fig.data[0].y[0]
    assert quality_fig.data[0].customdata[0][3] == "passed_all_qc"


def test_fig_historical_sensitivity_plots_sorted_change_from_full_sample():
    rows = pd.DataFrame({
        "cenario": ["Amostra completa", "Sem GP A", "Sem Japão", "Sem GP C"],
        "gp_excluido": ["Nenhum", "GP A", "Japanese Grand Prix", "GP C"],
        "pares": [165, 155, 156, 160], "media_seg": [3.0, 3.5, 2.4, 3.1],
    })
    fig = fig_historical_sensitivity(rows)
    trace = fig.data[0]
    assert list(trace.x) == pytest.approx([-0.6, 0.5, 0.1])
    assert list(trace.y) == ["Sem Japão ★", "Sem GP A", "Sem GP C"]
    assert list(trace.text)[0].startswith("↓ redução")
    assert list(trace.text)[-1].startswith("↑ aumento")
    assert trace.cliponaxis is False
    assert fig.layout.margin.l >= 150 and fig.layout.margin.r >= 100
    assert "Média completa" in trace.hovertemplate
    assert "Média sem GP" in trace.hovertemplate
    assert "Variação vs completa" in trace.hovertemplate
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].x0 == 0
