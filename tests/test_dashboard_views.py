"""Testes de navegacao e estados vazios das views (src/dashboard/views.py).

Usam tabelas sinteticas e nao tocam nos CSVs históricos.
"""

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import pipeline
from src.dashboard.views import _quality_class_label

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def isolate_streamlit_cache():
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()
    st.session_state.clear()


def _run_app(synthetic_tables, monkeypatch):
    monkeypatch.setattr(pipeline, "load_data", lambda raw_dir: deepcopy(synthetic_tables))
    return AppTest.from_file(str(APP), default_timeout=45).run()


def test_historico_exibe_estado_vazio_sem_pares(synthetic_tables, monkeypatch):
    """Sintetico tem 2014 e 2015, nao 2013: nao ha pares comparaveis."""
    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio[0].set_value("Comparação Histórica 2013–2014")
    app.run()
    assert not app.exception
    assert not app.error
    # estado vazio localizado na area, sem interromper outras areas
    app.sidebar.radio[0].set_value("Visão Geral")
    app.run()
    assert not app.exception
    assert not app.error


def test_explorador_estado_vazio_local(synthetic_tables, monkeypatch):
    """Filtro sem intersecao exibe estado vazio somente no Explorador."""
    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio[0].set_value("Explorador de Corridas")
    app.run()
    anos_widget = [w for w in app.sidebar.multiselect if w.key == "f1_filter_years"][0]
    anos_widget.set_value([2014])
    corridas_widget = [w for w in app.sidebar.multiselect if w.key == "f1_filter_races"][0]
    corridas_widget.set_value([11])  # GP B (2015), fora de 2014 -> intersecao vazia
    app.run()
    assert not app.exception
    assert not app.error
    registros = [m.value for m in app.metric if m.label == "Registros"]
    assert registros == ["0"], f"esperado 0 registros, obteve {registros}"
    # outras areas continuam funcionando
    app.sidebar.radio[0].set_value("Visão Geral")
    app.run()
    assert not app.exception
    assert not app.error


def test_explorador_contexto_e_soma_do_mapa_de_calor(synthetic_tables, monkeypatch):
    """O contexto dos filtros e a soma das celulas acompanham o recorte aplicado."""
    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio[0].set_value("Explorador de Corridas")
    app.run()
    assert not app.exception
    assert not app.error

    captions = " ".join(c.value for c in app.caption)
    assert "Recorte aplicado" in captions
    assert "Soma das células: 6 registros" in captions

    # Reduzir o recorte reduz a soma das celulas na mesma medida.
    [w for w in app.sidebar.multiselect if w.key == "f1_filter_grids"][0].set_value([1, 2])
    app.run()
    assert not app.exception
    assert not app.error
    captions = " ".join(c.value for c in app.caption)
    assert "Soma das células: 4 registros" in captions
    registros = [m.value for m in app.metric if m.label == "Registros"]
    assert registros == ["4"]


def test_explorador_alterna_modo_do_mapa_de_calor(synthetic_tables, monkeypatch):
    """O mapa de calor alterna entre contagem absoluta e percentual por grid."""
    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio[0].set_value("Explorador de Corridas")
    app.run()
    assert not app.exception
    assert not app.error

    modos = app.radio(key="f1_heatmap_mode")
    assert modos.value == "Contagem absoluta"

    modos.set_value("Percentual por grid")
    app.run()
    assert not app.exception
    assert not app.error
    assert app.radio(key="f1_heatmap_mode").value == "Percentual por grid"
    captions = " ".join(c.value for c in app.caption)
    assert "não do total geral do recorte" in captions
    assert "Soma das células: 6 registros" in captions


def test_explorador_filtros_combinados_modo_e_reset(synthetic_tables, monkeypatch):
    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio[0].set_value("Explorador de Corridas")
    app.run()
    for key, values in {
        "f1_filter_years": [2014],
        "f1_filter_races": [10],
        "f1_filter_drivers": [1, 2],
        "f1_filter_constructors": [7],
        "f1_filter_grids": [1, 2],
    }.items():
        app.sidebar.multiselect(key=key).set_value(values)
    app.run()
    app.radio(key="f1_heatmap_mode").set_value("Percentual por grid")
    app.run()
    assert not app.exception
    assert not app.error
    assert [m.value for m in app.metric if m.label == "Registros"] == ["2"]
    captions = " ".join(c.value for c in app.caption)
    assert "Soma das células: 2 registros" in captions
    assert "Temporadas: 2014" in captions
    assert "Grid: P1, P2" in captions
    assert len(app.dataframe[-1].value) == 2
    assert [e.label for e in app.main.expander] == [
        "Tabela de frequências por grid", "Registros do recorte",
    ]
    assert [e.label for e in app.sidebar.expander] == ["Glossário educativo", "Avisos metodológicos"]

    app.sidebar.button(key="f1_reset_filters").click().run()
    assert not app.exception
    assert not app.error
    assert [m.value for m in app.metric if m.label == "Registros"] == ["6"]
    assert all(not widget.value for widget in app.sidebar.multiselect)
    assert app.sidebar.radio[0].value == "Explorador de Corridas"
    assert app.radio(key="f1_heatmap_mode").value == "Percentual por grid"


def test_overview_exibe_indicadores_e_labels(synthetic_tables, monkeypatch):
    """Visão Geral exibe os indicadores principais com rótulos intuitivos, sem exceção."""
    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio[0].set_value("Visão Geral")
    app.run()
    assert not app.exception
    assert not app.error
    labels = [m.label for m in app.metric]
    esperados = [
        "Registros analisados",
        "Corridas distintas",
        "Pilotos diferentes",
        "Construtoras diferentes",
        "Vitória largando em P1",
        "Pódio largando em P1",
    ]
    for esperado in esperados:
        assert esperado in labels, f"rótulo ausente: {esperado} (obteve {labels})"


def _tabelas_historicas_2013_2014():
    """Três pares validos (mesmo piloto, construtora e circuito) em 2013 e 2014."""
    return {
        "results": pd.DataFrame({
            "resultId": [201, 202, 203, 204, 205, 206],
            "raceId": [100, 100, 100, 101, 101, 101],
            "driverId": [1, 2, 3, 1, 2, 3],
            "constructorId": [7, 7, 8, 7, 7, 8],
            "statusId": [1, 1, 1, 1, 1, 1],
            "grid": [1, 2, 3, 1, 2, 3],
            "positionOrder": [1, 2, 3, 1, 2, 3],
            "position": ["1", "2", "3", "1", "2", "3"],
            "positionText": ["1", "2", "3", "1", "2", "3"],
            "points": [25, 18, 15, 25, 18, 15],
            "laps": [58, 58, 58, 56, 56, 56],
            "fastestLapTime": ["1:25.0", "1:26.0", "1:28.0", "1:27.0", "1:24.0", "1:31.0"],
            "fastestLapSpeed": [215, 214, 210, 213, 216, 209],
        }),
        "races": pd.DataFrame({
            "raceId": [100, 101], "year": [2013, 2014],
            "circuitId": [20, 20], "name": ["GP Japão", "GP Japão"],
        }),
        "drivers": pd.DataFrame({
            "driverId": [1, 2, 3], "forename": ["Ana", "Beto", "Caio"], "surname": ["Um", "Dois", "Três"],
        }),
        "constructors": pd.DataFrame({
            "constructorId": [7, 8], "name": ["Equipe A", "Equipe B"],
        }),
        "status": pd.DataFrame({
            "statusId": [1], "status": ["Finished"],
        }),
    }


def test_historico_exibe_indicadores_e_grafico_dedicado(monkeypatch):
    """Area historica exibe KPIs computados, historico dedicado sem rotulos de ML e tabela expansivel."""
    tabelas = _tabelas_historicas_2013_2014()
    app = _run_app(tabelas, monkeypatch)
    app.sidebar.radio[0].set_value("Comparação Histórica 2013–2014")
    app.run()
    assert not app.exception
    assert not app.error

    valores = {m.label: m.value for m in app.metric}
    assert valores.get("Pares comparáveis") == "3"
    assert valores.get("Diferença média") == "+1.000 s"
    assert valores.get("Diferença mediana") == "+2.000 s"

    subheaders = [s.value for s in app.subheader]
    assert "Comparação por Grande Prix" in subheaders
    assert "Distribuição das diferenças individuais" in subheaders
    assert "Conclusões e limitações" in subheaders
    assert "Sensibilidade — exclusão de um Grande Prêmio" in subheaders
    assert "Pares comparáveis — sem Japão" in [m.label for m in app.metric]
    assert "Diferença média — sem Japão" in [m.label for m in app.metric]
    assert "Diferença mediana — sem Japão" in [m.label for m in app.metric]
    assert any("Austrália 2014" in item.value for item in app.warning)

    # Grafico dedicado de diferencas de volta (verificado em unitario; aqui confirma renderizacao).
    captions = " ".join(c.value for c in app.caption)
    assert "positivo significa volta mais lenta em 2014" in captions
    assert "não isola pista, clima, estratégia nem regulamento" in captions

    # Sem rotulos da area de ML nessa secao.
    texto_area = " ".join(
        [c.value for c in app.caption]
        + [m.label for m in app.metric]
        + [m.value for m in app.metric]
        + [s.value for s in app.subheader]
    )
    assert "Erro em posições" not in texto_area
    assert "erros de previsão" not in texto_area

    # Tabela detalhada em secao expansivel, sem download (licenca pendente).
    assert "Tabela de pares comparáveis" in [e.label for e in app.main.expander]
    assert len(app.dataframe) >= 1

    # Amostra pequena (1 GP com 3 pares < 5) sinalizada, mediana nao ocultada.
    assert "menos de 5 pares comparáveis" in captions


def test_quality_class_presentation_translation_keeps_unknown_technical_codes():
    assert _quality_class_label("passed_all_qc") == "Qualidade aprovada"
    assert _quality_class_label("suspect") == "Valor suspeito"
    assert _quality_class_label("future_qc_class") == "future_qc_class"


def test_climate_area_loads_local_bundle_and_shows_scope(monkeypatch):
    """A sexta área renderiza dados locais e mantém o escopo de estação explícito."""
    from src.dashboard import data as dashboard_data

    tables = _tabelas_historicas_2013_2014()
    app = _run_app(tables, monkeypatch)
    app.sidebar.radio[0].set_value("Contexto Climático 2013–2014")
    app.run()

    assert not app.exception
    assert not app.error
    assert dashboard_data.climate_file_signature()
    assert {metric.label for metric in app.metric} >= {
        "GPs investigados", "Eventos com registros", "Observações",
        "Temperatura do ar QC-aceita", "Ponto de orvalho QC-aceito",
    }
    captions = " ".join(item.value for item in app.caption)
    infos = " ".join(item.value for item in app.info)
    assert "não é a largada real" in captions
    assert "90/90 observações numéricas disponíveis" in infos
    assert "não comprova as condições meteorológicas no circuito" in infos
    assert any("Temperatura do ar por Grande Prêmio" == item.value for item in app.subheader)


def test_reading_mode_persists_across_area_navigation(synthetic_tables, monkeypatch):
    app = _run_app(synthetic_tables, monkeypatch)
    mode = app.sidebar.selectbox(key="f1_reading_mode")
    assert mode.value == "Explicação simples"
    mode.set_value("Visão técnica")
    app.run()
    assert not app.exception
    assert app.sidebar.selectbox(key="f1_reading_mode").value == "Visão técnica"

    app.sidebar.radio(key="f1_area_radio").set_value("Explorador de Corridas")
    app.run()
    assert not app.exception
    assert app.sidebar.selectbox(key="f1_reading_mode").value == "Visão técnica"
    assert any("Leitura técnica" in item.value for item in app.markdown)

    app.sidebar.radio(key="f1_area_radio").set_value("Visão Geral")
    app.run()
    assert not app.exception
    assert app.sidebar.selectbox(key="f1_reading_mode").value == "Visão técnica"


def test_provenance_distinguishes_kaggle_distribution_from_upstream_rights(
    synthetic_tables, monkeypatch, tmp_path,
):
    from src.audit_data import create_audit_report
    from src.dashboard import data as dashboard_data

    report = create_audit_report(tmp_path, tables=deepcopy(synthetic_tables))
    provenance = report["provenance"]
    assert {"source", "source_url", "dataset_version", "extracted_at", "license", "redistribution_permission"} <= set(provenance)
    assert provenance["source"] is None
    assert provenance["source_url"] is None
    assert provenance["identified_distribution"] == "Formula 1 Race Data"
    assert provenance["distribution_publisher"] == "cjgdev"
    assert provenance["declared_license"] == "CC BY-NC-SA 4.0 (declarada na página do dataset no Kaggle)."
    assert "Fase 7C" in provenance["local_file_match_evidence"]
    assert "não refaz essa comparação automaticamente" in provenance["local_file_match_evidence"]
    monkeypatch.setattr(dashboard_data, "load_audit_report", lambda raw_dir, signature: report)

    app = _run_app(synthetic_tables, monkeypatch)
    app.sidebar.radio(key="f1_area_radio").set_value("Dados e Metodologia")
    app.run()

    assert not app.exception
    assert not app.error
    content = "\n".join(item.value for item in app.markdown)
    warnings = "\n".join(item.value for item in app.warning)
    assert "Formula 1 Race Data" in content
    assert "https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017" in content
    assert "Fonte primária anterior ao Kaggle: PENDENTE" in content
    assert "CC BY-NC-SA 4.0" in content
    assert "Auditoria da Fase 7C" in content
    assert "não confirmada" in content
    assert "nem confirmam os direitos" in warnings
