"""Apresentação do ML: interpretações e diagnósticos sem recalcular modelos."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import pipeline
from src.dashboard import data as dashboard_data
from src.dashboard.charts import fig_calibration, fig_error_distribution, fig_mae_comparison, fig_observed_vs_predicted, fig_precision_recall, fig_roc
from src.dashboard.views import _classification_interpretation, _ml_table, _ml_value, _regression_interpretation

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def clear_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()
    st.session_state.clear()


def test_regressao_interpreta_metricas_recebidas_sem_conclusao_estatica():
    refs = {
        "mae_previsao_posicao_igual_ao_grid": 3.0,
        "rmse_previsao_posicao_igual_ao_grid": 6.0,
        "r2_previsao_posicao_igual_ao_grid": 0.2,
    }
    text = _regression_interpretation({"mae": 4.0, "rmse": 5.0, "r2": 0.3}, refs)
    assert "MAE, a regressão não supera" in text
    assert "RMSE, a regressão supera" in text
    assert "R², a regressão supera" in text
    assert "superioridade geral" in text
    assert "4.000 contra 3.000" in text

    text = _regression_interpretation({"mae": 2.0, "rmse": None, "r2": float("nan")}, refs)
    assert "MAE, a regressão supera" in text
    assert "Comparação de RMSE com a referência grid: indisponível" in text
    assert "Comparação de R² com a referência grid: indisponível" in text
    assert _ml_value(float("nan")) == "Indisponível"


def test_classificacao_condiciona_mensagens_aos_resultados():
    clf = {"recall": 0.91, "precision": 0.46, "roc_auc": 0.94, "grid_ranking_auc": 0.94}
    text = _classification_interpretation(clf)
    assert "recall é alto" in text and "precisão é moderada" in text
    assert "coincide" in text
    assert "selecionado no treino" in text
    assert "Brier Score e Log Loss" in text

    text = _classification_interpretation({**clf, "recall": 0.35, "precision": 0.9, "grid_ranking_auc": 0.7})
    assert "recall é alto" not in text
    assert "coincide" not in text
    assert "difere do ranking" in text

    text = _classification_interpretation({"recall": None, "precision": float("nan"), "roc_auc": None})
    assert "indisponíveis" in text and "coincide" not in text
    assert "indisponível" in _classification_interpretation({"erro_classificador": "sem duas classes"})


def test_tabelas_ml_renomeiam_sem_mudar_valores():
    frame = _ml_table([{"temporada": "2016", "n": 27, "mae": 3.1, "rmse": 5.0, "r2": 0.2}])
    assert list(frame.columns) == ["Temporada", "Amostras", "MAE", "RMSE", "R²"]
    assert frame.iloc[0].tolist() == ["2016", 27, 3.1, 5.0, 0.2]
    nested = _ml_table([{"temporada": "2016", "support": {0: 7, 1: 3}, "calibration_bins": [{"count": 10}]}])
    assert nested.loc[0, "support"] == '{"0": 7, "1": 3}'
    assert nested.loc[0, "calibration_bins"] == '[{"count": 10}]'


def test_curva_calibracao_nao_plota_bin_vazio_e_mantem_contagem():
    bins = [
        {"bin": 1, "mean_predicted": 0.1, "fraction_positives": 0.2, "count": 12},
        {"bin": 2, "mean_predicted": 0.5, "fraction_positives": None, "count": 0},
        {"bin": 3, "mean_predicted": 0.8, "fraction_positives": 0.7, "count": 4},
    ]
    fig = fig_calibration(bins, title=None)
    assert fig.layout.title.text is None
    assert list(fig.data[0].x) == [0.1, 0.8]
    assert list(fig.data[0].y) == [0.2, 0.7]
    assert list(fig.data[0].customdata) == [12, 4]
    assert "Registros" in fig.data[0].hovertemplate
    assert len(fig.data) == 2  # pontos observados + diagonal de calibração perfeita


def test_titulos_internos_podem_ser_ocultados_sem_perder_referencias():
    assert fig_mae_comparison({"Modelo": 2.0}, title=None).layout.title.text is None
    observed = fig_observed_vs_predicted([1, 2], [1.3, 2.2], title=None)
    assert observed.layout.title.text is None
    assert observed.data[1].name == "Identidade (observado = previsto)"
    assert fig_error_distribution([0.1, -0.2], title=None).layout.xaxis.title.text == "Erro em posições"
    roc = fig_roc([0, 1], [0, 1], 0.5, title=None)
    pr = fig_precision_recall([0, 1], [1, .5], .7, .2, title=None)
    assert roc.layout.title.text is None and len(roc.data) == 2
    assert pr.layout.title.text is None and len(pr.data) == 2


def test_area_ml_indisponibilidade_sem_zero_inventado(synthetic_tables, monkeypatch):
    monkeypatch.setattr(pipeline, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    stats = {
        "regressao_linear": {"mae": 4.1, "rmse": 5.0, "r2": None, "per_season_teste": {}},
        "referencias": {"mae_previsao_posicao_igual_ao_grid": 3.9, "mae_previsao_constante_mediana": 5.0,
                        "rmse_previsao_posicao_igual_ao_grid": 5.5, "r2_previsao_posicao_igual_ao_grid": float("nan")},
        "classificacao_podio": {"erro_classificador": "treino sem duas classes", "confusion_matrix": None,
                                 "roc_auc": None, "pr_auc": None, "precision": None, "recall": None,
                                 "calibration_bins": [], "per_season_teste": {}},
        "validacao_temporal_treino": {"regressao": [], "classificacao": []},
    }
    arrays = {"finish_position_test": np.array([1, 2]), "regression_predictions_test": np.array([1.1, 2.2]),
              "regression_errors": np.array([-0.1, -0.2]), "classifier_probabilities_test": None,
              "target_test": np.array([1, 0])}
    monkeypatch.setattr(dashboard_data, "evaluate_model_experiment", lambda df, sig: (stats, arrays))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Laboratório de Machine Learning")
    app.run()
    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["R²: modelo"] == "Indisponível"
    assert metrics["Average Precision (AP)"] == "Indisponível"
    assert metrics["Bins ocupados"] == "Indisponível"
    assert metrics["MAE: regressão linear"] == "4.100"
    infos = " ".join(i.value for i in app.info)
    assert "Classificador indisponível" in infos
    assert "Matriz de confusão indisponível" in infos
    assert "Curva ROC indisponível" in infos
    assert "não supera a previsão = grid" in infos


def test_area_ml_matriz_tabelas_e_bins_acessiveis(synthetic_tables, monkeypatch):
    monkeypatch.setattr(pipeline, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    stats = {
        "regressao_linear": {"mae": 4.0, "rmse": 5.0, "r2": .3, "per_season_teste": {"2015": {"n": 10, "mae": 4.0, "rmse": 5.0, "r2": .3}}},
        "referencias": {"mae_previsao_posicao_igual_ao_grid": 3.0, "mae_previsao_constante_mediana": 5.0,
                        "rmse_previsao_posicao_igual_ao_grid": 6.0, "r2_previsao_posicao_igual_ao_grid": .2},
        "classificacao_podio": {"roc_auc": .9, "grid_ranking_auc": .9, "pr_auc": .7,
                                 "brier_score": .1, "log_loss": .3, "precision": .5, "recall": .9, "f1": .65,
                                 "threshold": .2, "confusion_matrix": [[3, 2], [0, 5]],
                                 "calibration_ece": .03, "calibration_mce": .1, "train_prevalence": .5,
                                 "calibration_bins": [{"bin": 1, "mean_predicted": .1, "fraction_positives": .2, "count": 10},
                                                      {"bin": 2, "mean_predicted": .7, "fraction_positives": None, "count": 0}],
                                 "per_season_teste": {"2015": {"n": 10, "roc_auc": .9}}},
        "validacao_temporal_treino": {"regressao": [], "classificacao": []},
    }
    arrays = {"finish_position_test": np.array([1, 2, 3, 4, 5, 6, 1, 2, 4, 5]),
              "regression_predictions_test": np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 1.1, 2.1, 4.1, 5.1]),
              "regression_errors": np.full(10, -.1), "classifier_probabilities_test": np.array([.9, .8, .7, .2, .3, .1, .9, .8, .4, .1]),
              "target_test": np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])}
    monkeypatch.setattr(dashboard_data, "evaluate_model_experiment", lambda df, sig: (stats, arrays))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.sidebar.radio[0].set_value("Laboratório de Machine Learning")
    app.run()
    assert not app.exception and not app.error
    assert {m.label: m.value for m in app.metric}["Average Precision (AP)"] == "0.700"
    assert [e.label for e in app.main.expander] == [
        "Diagnósticos da regressão: previsões e resíduos", "Temporalidade — regressão",
        "Registros por faixa de probabilidade", "Temporalidade — classificação",
    ]
    matrix = next(df.value for df in app.dataframe if list(df.value.index) == ["Observado: não pódio", "Observado: pódio"])
    assert matrix.loc["Observado: não pódio", "Previsto: pódio"] == 2
    assert matrix.loc["Observado: pódio", "Previsto: não pódio"] == 0
    assert "falsos positivos (pódio previsto, não ocorreu) 2" in " ".join(c.value for c in app.caption)
    bin_table = next(df.value for df in app.dataframe if "Probabilidade média" in df.value.columns)
    assert bin_table["Registros"].tolist() == [10, 0]
    assert pd.isna(bin_table.loc[1, "Fração observada"])
    reg_table = next(df.value for df in app.dataframe if "R²" in df.value.columns)
    assert list(reg_table.columns[:5]) == ["Temporada", "Amostras", "MAE", "RMSE", "R²"]
