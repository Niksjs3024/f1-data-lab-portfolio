"""Testes das explicações educativas e seus estados ausentes."""

import pandas as pd

from src.dashboard.education import (
    classification_explanation,
    format_pt_br,
    historical_explanation,
    overview_explanation,
    regression_explanation,
)


def test_overview_explanation_uses_counts_and_actual_p1_metrics():
    text = " ".join(overview_explanation(
        pd.DataFrame({"grid": [1], "corridas": [12], "vitoria_pct": [25.0], "podio_pct": [50.0]}),
        {"registros": 30, "eventos": 10},
    ))
    assert "Entre 12 largadas" in text
    assert "25,0% venceram" in text
    assert "50,0% chegaram ao pódio" in text
    assert "30 resultados em 10 corridas" in text
    assert "primeira posição registrada no grid" in text
    assert "frequências históricas observadas" in text


def test_overview_explanation_missing_p1_is_not_zero():
    text = " ".join(overview_explanation(pd.DataFrame(), {}))
    assert "frequência calculável" in text
    assert "0.0%" not in text


def test_historical_explanation_uses_sensitivity_values_and_sign():
    text = " ".join(historical_explanation(
        {"pares": 165, "media_diferenca_seg": 3.256, "mediana_diferenca_seg": 2.1},
        pd.DataFrame([
            {"cenario": "Amostra completa", "media_seg": 3.256},
            {"cenario": "Sem Japão", "media_seg": 2.476},
        ]),
    ))
    assert "165 pares" in text
    assert "+3,256 s" in text and "+2,100 s" in text
    assert "mais lenta em 2014" in text
    assert "mudou -0,780 s" in text
    assert text.count("0,780") == 1
    assert "inconsistência documentada na Austrália 2014" in text


def test_pt_br_formatter_uses_decimal_comma_and_thousands_dot():
    assert format_pt_br(12345.678, 2, " s") == "12.345,68 s"
    assert format_pt_br(-0.5, 1, "%") == "-0,5%"
    assert format_pt_br(float("nan"), 2) == "indisponível"


def test_ml_explanations_use_plain_language_and_percent_scale():
    regression = " ".join(regression_explanation(
        {"mae": 3.7748}, {"mae_previsao_posicao_igual_ao_grid": 3.6926},
    ))
    classification = " ".join(classification_explanation({
        "precision": 0.63, "recall": 0.75, "f1": 0.68,
        "roc_auc": 0.8, "pr_auc": 0.6, "grid_ranking_auc": 0.8,
    }))
    assert "posição final registrada" in regression
    assert "3,8 posições" in regression
    assert "regra simples baseada no grid teve erro um pouco menor: 3,7 posições" in regression
    assert "De cada 100 pódios reais" in classification
    assert "cerca de 75%" in classification
    assert "De cada 100 previsões de pódio" in classification
    assert "cerca de 63%" in classification
    assert "não são novos testes com exatamente 100 casos" in classification


def test_missing_ml_metrics_remain_unavailable_not_zero():
    regression = " ".join(regression_explanation({}, {}))
    classification = " ".join(classification_explanation({"erro_classificador": True}))
    assert "não está disponível" in regression
    assert "0.000" not in regression
    assert "não pôde ser avaliado" in classification
    assert "não significa zero" in classification
