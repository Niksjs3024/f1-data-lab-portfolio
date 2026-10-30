"""Dashboard de portfólio: execute `streamlit run app.py`.

Orquestrador do dashboard: configuração da página, tema, carregamento de dados,
navegação entre seis áreas e despacho condicional. As regras estatísticas
permanecem em src/pipeline.py, src/analysis.py e src/model.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard import (
    RAW_DIR,
    file_signature,
    load_app_data,
    apply_theme,
)
from src.dashboard.education import READING_MODES, render_glossary
from src.dashboard.views import (
    render_data_methodology,
    render_explorer,
    render_historical,
    render_climate,
    render_ml,
    render_overview,
)

st.set_page_config(page_title="F1 | Data Lab", page_icon="🏎️", layout="wide")
apply_theme()

try:
    signature = file_signature(RAW_DIR)
    all_data: pd.DataFrame = load_app_data(str(RAW_DIR), signature)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Não foi possível carregar os dados: {exc}")
    st.stop()

st.title("🏎️ F1 | Data Lab")
st.caption("Projeto pessoal de análise de dados · Python + Pandas + Estatística + Machine Learning")

AREAS = [
    ("Visão Geral", "visao_geral"),
    ("Explorador de Corridas", "explorador"),
    ("Comparação Histórica 2013–2014", "historico"),
    ("Laboratório de Machine Learning", "ml"),
    ("Dados e Metodologia", "dados"),
    ("Contexto Climático 2013–2014", "clima"),
]
LABEL_TO_KEY = {label: key for label, key in AREAS}

if "f1_area" not in st.session_state:
    st.session_state.f1_area = "visao_geral"

with st.sidebar:
    st.header("Navegação")
    reading_mode = st.selectbox("Entenda os resultados", READING_MODES, key="f1_reading_mode")
    choice = st.radio("Área de análise", [label for label, _ in AREAS], key="f1_area_radio")
    render_glossary()
    area_key = LABEL_TO_KEY[choice]
    if st.session_state.f1_area != area_key:
        st.session_state.f1_area = area_key
    st.divider()

if area_key == "visao_geral":
    render_overview(all_data, reading_mode)
elif area_key == "explorador":
    render_explorer(all_data, reading_mode)
elif area_key == "historico":
    render_historical(all_data, reading_mode)
elif area_key == "ml":
    render_ml(all_data, signature, reading_mode)
elif area_key == "dados":
    render_data_methodology(all_data, signature, reading_mode)
elif area_key == "clima":
    render_climate(reading_mode)
else:
    st.error("Área desconhecida.")

# Avisos metodológicos ficam abaixo dos controles da área selecionada, para que
# os filtros do Explorador apareçam imediatamente após a navegação.
with st.sidebar:
    with st.expander("Avisos metodológicos", expanded=False):
        st.caption("Base histórica fornecida com o projeto. Frequências são observadas, não previsões.")
        st.caption("Grid é a posição efetiva de largada registrada, que pode diferir da classificação oficial.")
        st.caption("Origem e licença dos CSVs: pendentes de documentação. Uso somente local.")
