"""Smoke do app real com tabelas sintéticas, sem depender dos CSVs históricos.

Contratos preservados da versão anterior:
- O aplicativo abre sem exceções nem erros não tratados.
- Título e métricas são identificados por rótulo, não por posição.
- Filtros exploratórios alteram somente o Explorador.
- Grid duplicado é warning não bloqueante e mantém todos os registros.
- Falha de contrato é apresentada como mensagem, sem exceção não tratada.
- Cache de dados é isolado entre testes.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import pipeline

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def isolate_streamlit_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()
    # Isola o estado da sessão entre testes, pois o AppTest compartilha o
    # st.session_state global do processo e a navegação/rádio/filtros gravam nele.
    st.session_state.clear()


def _metric_by_label(app, label):
    for metric in app.metric:
        if metric.label == label:
            return metric
    return None


def _sidebar_multiselect_by_key(app, key):
    matches = [widget for widget in app.sidebar.multiselect if widget.key == key]
    assert matches, f"widget {key} não encontrado na sidebar"
    return matches[0]


@pytest.mark.parametrize("duplicate_grid", [False, True])
def test_app_runs_and_filters_with_compatible_validation(synthetic_tables, monkeypatch, duplicate_grid):
    if duplicate_grid:
        synthetic_tables["results"].loc[0, "grid"] = 2
    monkeypatch.setattr(pipeline, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == "🏎️ F1 | Data Lab"

    registros = _metric_by_label(app, "Registros analisados")
    assert registros is not None and registros.value == "6"

    # A navegação muda a área sem recarregar nem treinar modelos.
    app.sidebar.radio[0].set_value("Explorador de Corridas")
    app.run()
    assert not app.exception
    assert not app.error

    registros = _metric_by_label(app, "Registros")
    assert registros is not None and registros.value == "6"

    # Filtro por grid P2: 2 registros, ou 3 quando P17 não se aplica (duplicata em grid 2).
    grid_widget = _sidebar_multiselect_by_key(app, "f1_filter_grids")
    grid_widget.set_value([2])
    app.run()
    assert not app.exception
    assert not app.error
    registros = _metric_by_label(app, "Registros")
    expected = "3" if duplicate_grid else "2"
    assert registros is not None and registros.value == expected

    # Redefinição restaura o recorte completo.
    reset = [b for b in app.sidebar.button if b.key == "f1_reset_filters"]
    assert reset
    reset[0].click()
    app.run()
    assert not app.exception
    assert not app.error
    registros = _metric_by_label(app, "Registros")
    assert registros is not None and registros.value == "6"


def test_app_displays_contract_failure_without_uncaught_exception(synthetic_tables, monkeypatch):
    synthetic_tables["results"].loc[0, "driverId"] = 999
    monkeypatch.setattr(pipeline, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    messages = " ".join(error.value for error in app.error)
    assert "Não foi possível carregar os dados" in messages


def test_app_navigates_to_all_areas(synthetic_tables, monkeypatch):
    monkeypatch.setattr(pipeline, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    assert not app.error

    for area in [
        "Explorador de Corridas",
        "Comparação Histórica 2013–2014",
        "Laboratório de Machine Learning",
        "Dados e Metodologia",
    ]:
        app.sidebar.radio[0].set_value(area)
        app.run()
        assert not app.exception, f"exceção na área {area}"
        assert not app.error, f"erro na área {area}"
