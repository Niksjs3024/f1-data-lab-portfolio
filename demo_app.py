
import streamlit as st

st.set_page_config(
    page_title="F1 Data Lab | Portfólio",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = "https://github.com/Niksjs3024/f1-data-lab-portfolio"

st.markdown("""
<style>
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.f1-hero {
    background: linear-gradient(120deg, #251313, #141414 65%, #1c2028);
    border: 1px solid #503030;
    border-left: 5px solid #e23d31;
    padding: 30px;
    border-radius: 14px;
    margin-bottom: 22px;
}
.f1-hero h1 {
    margin: 0 0 12px 0;
    font-size: 2.5rem;
    color: #f5f5f5;
}
.f1-hero p {
    color: #d1d1d1;
    line-height: 1.7;
    margin-bottom: 0;
}
.f1-kicker {
    color: #ff8279;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

AREAS = {
    "Visão Geral": {
        "subtitulo": "Exploração de desempenho histórico na Fórmula 1",
        "simples": (
            "A versão completa investiga como as posições de largada "
            "se relacionam com os resultados das corridas."
        ),
        "tecnico": (
            "Análise de frequências condicionadas à posição de grid, "
            "com integração de resultados, pilotos, equipes e corridas. "
            "O recorte padrão utiliza temporadas de 2004 a 2017 e grid positivo."
        ),
        "metodos": [
            "Integração de tabelas por identificadores",
            "Tratamento e validação dos registros",
            "Cálculo de frequências empíricas",
        ],
        "limite": (
            "Frequências observadas não representam causalidade "
            "nem garantem resultados futuros."
        ),
    },
    "Explorador de Corridas": {
        "subtitulo": "Análise exploratória com filtros",
        "simples": (
            "O dashboard original permite investigar resultados "
            "de diferentes temporadas, corridas, pilotos e equipes."
        ),
        "tecnico": (
            "Filtros aplicados sobre dataset integrado, com seleção "
            "por temporada, race_id, driver_id, constructor_id e grid. "
            "As visualizações incluem tabelas, frequências e mapas de calor."
        ),
        "metodos": [
            "Filtros dinâmicos com Streamlit",
            "Agregações e segmentação com Pandas",
            "Visualizações orientadas aos recortes selecionados",
        ],
        "limite": (
            "Os filtros exploratórios não modificam a base "
            "nem o período de avaliação dos modelos."
        ),
    },
    "Comparação Histórica 2013–2014": {
        "subtitulo": "Pareamento e análise de sensibilidade",
        "simples": (
            "O estudo compara tempos de volta rápida entre "
            "2013 e 2014 para combinações equivalentes."
        ),
        "tecnico": (
            "Pareamento por piloto, construtora e circuito, "
            "com diferenças de tempos e análises de sensibilidade. "
            "A documentação registra limitações de reconciliação "
            "com resultados oficiais."
        ),
        "metodos": [
            "Pareamento histórico entre temporadas",
            "Estatísticas descritivas",
            "Análise de sensibilidade",
        ],
        "limite": (
            "A comparação não identifica isoladamente efeitos "
            "de regulamento, clima, pista ou estratégia."
        ),
    },
    "Laboratório de Machine Learning": {
        "subtitulo": "Modelagem, baselines e validação temporal",
        "simples": (
            "O projeto compara modelos com alternativas simples "
            "para verificar se o uso de Machine Learning realmente agrega valor."
        ),
        "tecnico": (
            "Experimentos de regressão e classificação usando grid "
            "como variável preditora. Treinamento em 2004–2014 "
            "e teste temporal reservado em 2015–2017. "
            "A regressão não superou o baseline baseado no grid em MAE."
        ),
        "metodos": [
            "Regressão e classificação com scikit-learn",
            "Separação temporal de treino e teste",
            "Comparação com baselines e métricas de avaliação",
        ],
        "limite": (
            "Os modelos são exploratórios e não devem ser "
            "interpretados como previsões confiáveis de corridas futuras."
        ),
    },
    "Dados e Metodologia": {
        "subtitulo": "Qualidade, contratos e rastreabilidade",
        "simples": (
            "A documentação explica de onde vêm os dados, "
            "como são preparados e quais limitações precisam ser consideradas."
        ),
        "tecnico": (
            "Pipeline Python para leitura, integração e validação de "
            "cinco CSVs históricos. Auditorias e contratos de dados "
            "documentam critérios de qualidade e proveniência."
        ),
        "metodos": [
            "Contratos e verificações de consistência",
            "Auditorias de dados",
            "Testes automatizados com pytest",
        ],
        "limite": (
            "A integridade dos arquivos foi verificada, mas isso "
            "não comprova autorização para redistribuição."
        ),
    },
    "Contexto Climático 2013–2014": {
        "subtitulo": "Observações meteorológicas e evidências",
        "simples": (
            "O projeto investiga informações climáticas regionais "
            "e registros documentais relacionados a determinadas corridas."
        ),
        "tecnico": (
            "Integração de observações de estações NOAA, "
            "classificações de qualidade instrumental e evidências "
            "documentais de eventos, mantendo essas fontes separadas."
        ),
        "metodos": [
            "Consulta e organização de observações NOAA",
            "Interpretação de flags de qualidade",
            "Documentação de distância, horários e representatividade",
        ],
        "limite": (
            "Dados de estações próximas não equivalem a medições "
            "feitas no circuito durante a corrida."
        ),
    },
}

ETAPAS = [
    ("01", "Fontes", "CSVs históricos, NOAA e referências documentais"),
    ("02", "Validação", "Verificação de estrutura, consistência e qualidade"),
    ("03", "Pipeline", "Integração, limpeza e preparação com Python"),
    ("04", "Análise", "Estatística descritiva e estudos históricos"),
    ("05", "Modelagem", "Machine Learning, baselines e avaliação temporal"),
    ("06", "Interface", "Visualização e navegação em Streamlit"),
]

with st.sidebar:
    st.header("🏎️ F1 Data Lab")
    st.caption("Demonstração pública do projeto")

    area = st.radio("Área de análise", list(AREAS))
    modo = st.selectbox(
        "Modo de leitura",
        ["Explicação simples", "Visão técnica"],
    )

    st.divider()
    st.link_button("Ver código no GitHub", GITHUB_URL)
    st.caption("Demonstração sem redistribuição dos datasets.")

st.markdown("""
<div class="f1-hero">
    <div class="f1-kicker">DATA ANALYTICS · PYTHON · MACHINE LEARNING</div>
    <h1>🏎️ F1 Data Lab</h1>
    <p>
        Um projeto de análise histórica da Fórmula 1 que reúne
        engenharia e qualidade de dados, estatística,
        Machine Learning e visualização interativa.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.metric("Testes automatizados", "331")
        st.caption("Aprovados na versão local completa")

with c2:
    with st.container(border=True):
        st.metric("Áreas de análise", "6")
        st.caption("Verificadas no dashboard original")

with c3:
    with st.container(border=True):
        st.metric("Python de referência", "3.12")
        st.caption("Ambiente independente validado")

st.info(
    "Esta é uma demonstração informativa. "
    "As análises completas dependem de dados locais não distribuídos aqui."
)

aba_area, aba_arquitetura, aba_validacao = st.tabs([
    "Explorar o projeto",
    "Arquitetura",
    "Validação e limites",
])

with aba_area:
    info = AREAS[area]

    st.subheader(area)
    st.caption(info["subtitulo"])

    with st.container(border=True):
        st.markdown("#### O que foi desenvolvido")
        st.write(
            info["simples"]
            if modo == "Explicação simples"
            else info["tecnico"]
        )

    st.markdown("#### Métodos e tecnologias")
    for metodo in info["metodos"]:
        st.markdown(f"- {metodo}")

    with st.container(border=True):
        st.markdown("**Limitação importante**")
        st.write(info["limite"])

with aba_arquitetura:
    st.subheader("Arquitetura do projeto")
    st.write(
        "O fluxo abaixo representa a estrutura real do F1 Data Lab. "
        "A demonstração pública não executa o pipeline analítico."
    )

    for inicio in (0, 3):
        colunas = st.columns(3)

        for coluna, etapa in zip(colunas, ETAPAS[inicio:inicio + 3]):
            numero, titulo, descricao = etapa

            with coluna:
                with st.container(border=True):
                    st.caption(f"ETAPA {numero}")
                    st.markdown(f"#### {titulo}")
                    st.write(descricao)

    st.markdown("#### Organização técnica")
    st.code(
        "app.py                  # Dashboard analítico original\n"
        "src/pipeline.py         # Ingestão e integração\n"
        "src/data_quality.py     # Qualidade e validação\n"
        "src/analysis.py         # Análises estatísticas\n"
        "src/model.py            # Modelos e baselines\n"
        "src/dashboard/          # Interface e visualizações\n"
        "tests/                  # Testes automatizados",
        language="text",
    )

with aba_validacao:
    st.subheader("Validação técnica")

    st.success(
        "A aplicação analítica completa foi validada localmente "
        "em uma instalação Python independente."
    )

    st.markdown(
        "- 331 testes automatizados aprovados.\n"
        "- Seis áreas verificadas manualmente no Streamlit.\n"
        "- Integridade de arquivos de dados e figuras conferida por SHA-256.\n"
        "- Código organizado em módulos com responsabilidades separadas."
    )

    st.markdown("#### Limites desta demonstração")
    st.warning(
        "Não há carregamento de CSVs históricos ou climáticos nesta página. "
        "As informações descrevem a metodologia e os resultados "
        "da validação técnica local."
    )

    st.write(
        "A proveniência dos dados, as condições de uso e as "
        "restrições de redistribuição estão documentadas no repositório."
    )

    st.link_button(
        "Consultar documentação no GitHub",
        GITHUB_URL + "/blob/main/docs/PROVENIENCIA_E_LICENCAS.md",
    )

st.divider()
st.caption(
    "F1 Data Lab · Projeto pessoal de portfólio. "
    "Sem afiliação ou endosso da Fórmula 1, FIA, equipes ou pilotos."
)
