"""Tokens visuais, template Plotly e CSS do F1 Data Lab.

A identidade visual usa fundo grafite/preto com vermelho como acento de
interface. A paleta categórica validada para superfícies escuras é:

- azul #3987e5
- laranja #d95926
- aqua #199e70

O vermelho institucional permanece como acento e nunca é usado como quarta
série categórica sem nova validação de contraste e separação de cores.
"""

from __future__ import annotations

SURFACE_DARK = "#0d0d0d"      # plano de fundo da página
SURFACE_PANEL = "#1a1a19"     # superfície de cards e gráficos
SURFACE_BORDER = "#2c2c2a"    # hairline de separação
INK_PRIMARY = "#f0efec"       # texto principal
INK_SECONDARY = "#c3c2b7"     # texto secundário
INK_MUTED = "#898781"         # eixos, rótulos auxiliares
ACCENT_RED = "#e23d31"        # destaque de interface, nunca de dados sozinho

PALETTE_CATEGORICAL = ["#3987e5", "#d95926", "#199e70"]
PALETTE_SEQUENTIAL = ["#16324f", "#2a78d6", "#3987e5", "#5aa9f0", "#86b6ef"]
PALETTE_DIVERGING = ["#d95926", "#52514e", "#3987e5"]

GRID_LINE_COLOR = "#2c2c2a"
AXIS_LINE_COLOR = "#383835"

CSS = f"""
<style>
.block-container {{
    padding-top: 1.4rem;
    max-width: 1400px;
}}
div[data-testid="stMetric"] {{
    border: 1px solid {SURFACE_BORDER};
    border-radius: 12px;
    padding: 12px 14px;
    background: #161615;
}}
section[data-testid="stSidebar"] div[data-testid="stMetric"] {{
    background: #161615;
}}
h1, h2, h3 {{
    letter-spacing: -.025em;
    color: {INK_PRIMARY};
}}
.scope-note {{
    color: {INK_MUTED};
    font-size: 0.85rem;
    margin-top: -0.35rem;
    margin-bottom: 1rem;
}}
.metric-helper {{
    color: {INK_SECONDARY};
    font-size: 0.78rem;
    margin-top: 0.2rem;
}}
.spotlight-title {{
    color: {ACCENT_RED};
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}}
.empty-state {{
    border: 1px dashed {SURFACE_BORDER};
    border-radius: 12px;
    padding: 2rem 1rem;
    text-align: center;
    color: {INK_SECONDARY};
    background: #161615;
}}
.dataframe thead th {{
    position: sticky;
    top: 0;
    background: {SURFACE_PANEL};
    z-index: 1;
}}
</style>
"""


def apply_theme() -> None:
    """Injeta o CSS estático mínimo e tokens da identidade visual."""
    import streamlit as st

    st.markdown(CSS, unsafe_allow_html=True)


def plotly_layout(base: dict | None = None) -> dict:
    """Retorna layout escuro consistente para figuras Plotly."""
    layout = {
        "paper_bgcolor": SURFACE_PANEL,
        "plot_bgcolor": SURFACE_PANEL,
        "font": {
            "color": INK_SECONDARY,
            "family": "system-ui, -apple-system, \"Segoe UI\", sans-serif",
            "size": 12,
        },
        "xaxis": {
            "gridcolor": GRID_LINE_COLOR,
            "linecolor": AXIS_LINE_COLOR,
            "tickcolor": INK_MUTED,
            "title": {"font": {"color": INK_MUTED}},
        },
        "yaxis": {
            "gridcolor": GRID_LINE_COLOR,
            "linecolor": AXIS_LINE_COLOR,
            "tickcolor": INK_MUTED,
            "title": {"font": {"color": INK_MUTED}},
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": SURFACE_BORDER,
            "font": {"color": INK_SECONDARY},
        },
        "margin": {"l": 60, "r": 20, "t": 40, "b": 60},
        "hovermode": "closest",
    }
    if base:
        layout.update(base)
    return layout


def scope_note(text: str) -> str:
    """Monta aviso de escopo de uma área de análise."""
    return f'<p class="scope-note">{text}</p>'
