"""Figuras Plotly puras: recebem dados computados e devolvem figuras.

Sem Streamlit, sem treinamento, sem gravação em disco. As regras estatísticas
permanecem em src/analysis.py e src/model.py; aqui só há apresentação.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import (
    ACCENT_RED,
    AXIS_LINE_COLOR,
    GRID_LINE_COLOR,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    PALETTE_CATEGORICAL,
    PALETTE_SEQUENTIAL,
    plotly_layout,
)


def _base_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**plotly_layout())
    return fig


def fig_grid_probabilities(proba: pd.DataFrame, grids: list[int] | None = None) -> go.Figure:
    """Vitória, pódio e Top 10 observados por posição de grid (0 a 100%)."""
    data = proba.copy()
    if grids:
        data = data[data["grid"].isin(grids)]
    if data.empty:
        fig = _base_figure()
        fig.update_layout(title="Sem registros para os grids selecionados")
        return fig

    # Critério de apresentação (não estatístico): grids com poucos registros analisados
    # recebem marcador mais discreto. Percentuais nunca são alterados nem ocultados.
    SMALL_SAMPLE_THRESHOLD = 20
    data["_small"] = data["corridas"] < SMALL_SAMPLE_THRESHOLD

    fig = _base_figure()
    series = [
        ("vitoria_pct", "Vitória", PALETTE_CATEGORICAL[0], "solid"),
        ("podio_pct", "Pódio", PALETTE_CATEGORICAL[1], "dash"),
        ("top10_pct", "Top 10", PALETTE_CATEGORICAL[2], "dot"),
    ]
    for column, label, color, dash in series:
        sizes = []
        opacities = []
        for _, row in data.iterrows():
            n = int(row["corridas"])
            sizes.append(6 if n >= SMALL_SAMPLE_THRESHOLD else 4)
            opacities.append(1.0 if n >= SMALL_SAMPLE_THRESHOLD else 0.55)
        fig.add_trace(
            go.Scatter(
                x=data["grid"],
                y=data[column],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 2, "dash": dash},
                marker={
                    "size": sizes,
                    "symbol": "circle",
                    "line": {"width": 1, "color": "#0d0d0d"},
                    "opacity": opacities,
                },
                customdata=data["corridas"],
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Grid: %{x}<br>"
                    "%{y:.2f}%<br>"
                    "Registros analisados neste grid: %{customdata}<extra></extra>"
                ),
            )
        )
    # O título da seção é responsabilidade da view (st.subheader); aqui só há
    # título quando a figura precisa comunicar estado vazio.
    fig.update_layout(
        xaxis_title="Posição de largada (grid)",
        yaxis_title="% dos resultados desse grid",
        yaxis={"range": [0, 105], "ticksuffix": "%", **plotly_layout()["yaxis"]},
        xaxis={**plotly_layout()["xaxis"]},
        legend={"title": None},
    )
    return fig


def fig_heatmap(counts: pd.DataFrame, mode: str = "count") -> go.Figure:
    """Matriz grid (linhas) × ordenação final registrada (colunas).

    `mode="count"` mostra os registros de cada célula; `mode="percent"` mostra
    o percentual condicionado ao grid, isto é, sobre o total de registros
    daquela posição de largada (denominador = total da linha) — nunca sobre o
    total geral do recorte. O domínio vem dos dados observados; não há corte
    artificial em P20.

    A variação entre posição de largada e ordem final registrada não mede
    número de ultrapassagens.
    """
    fig = _base_figure()
    if counts.empty:
        fig.update_layout(title="Sem registros para o recorte escolhido")
        return fig

    # A regra de normalização por linha vive em data.heatmap_percentages para
    # não existir uma segunda implementação de denominador na camada gráfica.
    from .data import heatmap_percentages

    percentages = heatmap_percentages(counts)
    cell_counts = counts.astype(int).values.tolist()
    row_totals = counts.sum(axis=1).astype(int).tolist()
    customdata = [
        [[cell_counts[i][j], row_totals[i], percentages.iat[i, j]]
         for j in range(len(counts.columns))]
        for i in range(len(counts.index))
    ]
    hovertemplate = (
        "Posição de largada: P%{y}<br>"
        "Ordem final registrada: P%{x}<br>"
        "Registros nesta célula: %{customdata[0]:.0f}<br>"
        "%{customdata[2]:.2f}% dos registros desse grid<br>"
        "Denominador (registros nesse grid): %{customdata[1]:.0f}<extra></extra>"
    )

    if mode == "percent":
        z = percentages.values
        colorbar = {"title": "% do grid", "ticksuffix": "%"}
    else:
        z = counts.values
        colorbar = {"title": "Registros"}

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=[str(c) for c in counts.columns],
            y=[str(r) for r in counts.index],
            colorscale=PALETTE_SEQUENTIAL,
            colorbar=colorbar,
            zmin=0,
            zmax=100 if mode == "percent" else None,
            hoverongaps=False,
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(
        xaxis_title="Ordem final registrada (positionOrder)",
        yaxis_title="Posição de largada (grid)",
        yaxis={"autorange": "reversed"},
    )
    return fig


def fig_lap_time_distribution(differences) -> go.Figure:
    """Histograma das diferenças 2014 − 2013 do tempo de volta mais rápida (segundos).

    Recebe exclusivamente diferenças de tempo em segundos. Todos os valores
    válidos entram nas barras, extremos incluídos, sem descarte arbitrário.
    """
    fig = _base_figure()
    values = [float(v) for v in differences if pd.notna(v)]
    if not values:
        fig.update_layout(title="Sem diferenças de tempo de volta para exibir")
        return fig

    fig.add_trace(
        go.Histogram(
            x=values,
            name="Pares",
            marker={"color": PALETTE_CATEGORICAL[0], "opacity": 0.75},
            hovertemplate="Diferença (s): %{x}<br>Quantidade de pares: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=0,
        line={"color": ACCENT_RED, "width": 1, "dash": "dash"},
        annotation_text="Zero (sem mudança)",
        annotation_position="top left",
        annotation_font={"color": ACCENT_RED},
    )
    fig.update_layout(
        title="Distribuição das diferenças de tempo de volta",
        xaxis_title="Diferença de tempo (segundos)",
        yaxis_title="Quantidade de pares",
        bargap=0.1,
    )
    return fig


def fig_observed_vs_predicted(y_true, y_pred, *, title: str | None = "Ordenação final observada versus prevista (regressão linear)") -> go.Figure:
    """Observado versus previsto com diagonal de identidade."""
    fig = _base_figure()
    if len(y_true) == 0:
        fig.update_layout(title="Sem previsões para exibir")
        return fig

    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Resultado",
            marker={"color": PALETTE_CATEGORICAL[0], "size": 6, "opacity": 0.7},
            hovertemplate="Observado: %{x}<br>Previsto: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="Identidade (observado = previsto)",
            line={"color": ACCENT_RED, "width": 2, "dash": "dash"},
            hovertemplate="Identidade: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Observado (positionOrder)",
        yaxis_title="Previsto (posições)",
        xaxis={"range": [lo, hi], **plotly_layout()["xaxis"]},
        yaxis={"range": [lo, hi], **plotly_layout()["yaxis"]},
    )
    return fig


def fig_error_distribution(errors, *, title: str | None = "Distribuição dos erros de previsão (observado − previsto)") -> go.Figure:
    """Distribuição dos resíduos: observado menos previsto."""
    fig = _base_figure()
    if len(errors) == 0:
        fig.update_layout(title="Sem erros para exibir")
        return fig
    fig.add_trace(
        go.Histogram(
            x=errors,
            name="Resíduo",
            marker={"color": PALETTE_CATEGORICAL[0]},
            nbinsx=30,
            hovertemplate="Erro: %{x}<br>Contagem: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Erro em posições",
        yaxis_title="Contagem",
    )
    return fig


def fig_mae_comparison(values: dict[str, float], *, title: str | None = "Erro médio absoluto: regressão linear versus referências") -> go.Figure:
    """MAE do modelo versus referências (menor é melhor)."""
    fig = _base_figure()
    labels = list(values.keys())
    nums = [values[k] for k in labels]
    colors = [ACCENT_RED if k.startswith("Referência") else PALETTE_CATEGORICAL[0] for k in labels]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=nums,
            name="MAE",
            marker={"color": colors},
            hovertemplate="%{x}: %{y:.3f} posições<extra></extra>",
            text=[f"{v:.3f}" for v in nums],
            textposition="outside",
            textfont={"color": INK_PRIMARY},
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="MAE (posições; menor é melhor)",
        xaxis={"tickangle": -15},
        uniformtext={"mode": "hide", "minsize": 8},
    )
    return fig


def fig_roc(fpr, tpr, auc_value, *, title: str | None = "Curva ROC — classificação de pódio") -> go.Figure:
    """Curva ROC com linha de referência do classificador aleatório."""
    fig = _base_figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="Regressão logística",
            line={"color": PALETTE_CATEGORICAL[0], "width": 2},
            hovertemplate="Taxa de falso positivo: %{x:.3f}<br>Taxa de verdadeiro positivo: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Classificador aleatório",
            line={"color": INK_MUTED, "width": 1, "dash": "dash"},
            hovertemplate="Referência aleatória<extra></extra>",
        )
    )
    if title is not None and auc_value is not None:
        title += f"  (ROC AUC = {auc_value:.3f})"
    fig.update_layout(
        title=title,
        xaxis_title="Taxa de falso positivo",
        yaxis_title="Taxa de verdadeiro positivo",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return fig


def fig_precision_recall(recall, precision, ap_value, prevalence, *, title: str | None = "Curva precisão–recall — classificação de pódio") -> go.Figure:
    """Curva precisão–recall com linha de referência da probabilidade constante.

    `ap_value` é o Average Precision já computado; não se usa integração
    trapezoidal da curva desenhada.
    """
    fig = _base_figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Regressão logística",
            line={"color": PALETTE_CATEGORICAL[0], "width": 2},
            hovertemplate="Recall: %{x:.3f}<br>Precisão: %{y:.3f}<extra></extra>",
        )
    )
    if prevalence is not None:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[prevalence, prevalence],
                mode="lines",
                name="Probabilidade constante (treino)",
                line={"color": INK_MUTED, "width": 1, "dash": "dash"},
                hovertemplate=f"Prevalência constante: {prevalence:.3f}<extra></extra>",
            )
        )
    if title is not None and ap_value is not None:
        title += f"  (Average Precision = {ap_value:.3f})"
    fig.update_layout(
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precisão",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return fig


def fig_calibration(bins, *, title: str | None = "Curva de calibração do classificador de pódio") -> go.Figure:
    """Curva de calibração: fração observada versus probabilidade prevista."""
    fig = _base_figure()
    if not bins:
        fig.update_layout(title="Sem bins de calibração para exibir")
        return fig

    occupied = [b for b in bins if b.get("count", 0) and pd.notna(b.get("fraction_positives"))]
    means = [b["mean_predicted"] for b in occupied]
    fracs = [b["fraction_positives"] for b in occupied]
    counts = [b["count"] for b in occupied]

    fig.add_trace(
        go.Scatter(
            x=means,
            y=fracs,
            mode="lines+markers",
            name="Calibração",
            line={"color": PALETTE_CATEGORICAL[0], "width": 2},
            marker={"size": 8},
            customdata=counts,
            hovertemplate="Prob. prevista: %{x:.2f}<br>Fração observada: %{y}<br>Registros: %{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Calibração perfeita",
            line={"color": INK_MUTED, "width": 1, "dash": "dash"},
            hovertemplate="Referência<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Probabilidade prevista",
        yaxis_title="Fração observada de pódios",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    return fig


def fig_temporal(rows: list[dict], metric: str, title: str) -> go.Figure:
    """Métricas por temporada do teste ou validação progressiva."""
    fig = _base_figure()
    if not rows:
        fig.update_layout(title="Sem dados temporais para exibir")
        return fig

    labels = [str(r.get("ano", r.get("ano_validacao", ""))) for r in rows]
    values = [r.get(metric) for r in rows]
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            name=title,
            line={"color": PALETTE_CATEGORICAL[0], "width": 2},
            marker={"size": 7},
            hovertemplate="Ano: %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title="Temporada", yaxis_title=metric)
    return fig


def fig_delta_historical(
    rows: list[dict], value_col: str, label_col: str, title: str | None = None
) -> go.Figure:
    """Barras horizontais da diferença mediana 2014 menos 2013 por evento.

    `n_pares` em cada linha controla a opacidade (amostras pequenas ficam mais
    discretas, nunca ocultas) e aparece no hover.
    """
    fig = _base_figure()
    if not rows:
        fig.update_layout(title="Sem pares para exibir")
        return fig

    labels = [str(r.get(label_col, "")) for r in rows]
    values = [r.get(value_col) for r in rows]
    colors = [ACCENT_RED if (v is not None and v > 0) else PALETTE_CATEGORICAL[0] for v in values]
    opacity = [
        0.6 if (r.get("n_pares") is not None and int(r.get("n_pares")) < 5) else 1.0
        for r in rows
    ]
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "opacity": opacity},
            customdata=[[int(r.get("n_pares", 0))] for r in rows],
            hovertemplate="%{y}: %{x:.3f} s (%{customdata[0]} pares comparáveis)<extra></extra>",
            text=[f"{v:+.3f}" if v is not None else "—" for v in values],
            textposition="outside",
            textfont={"color": INK_PRIMARY},
        )
    )
    layout_updates = {
        "xaxis_title": "Diferença 2014 − 2013 (segundos; positivo = mais lento em 2014)",
        "yaxis": {"autorange": "reversed"},
        "uniformtext": {"mode": "hide", "minsize": 8},
    }
    if title:
        layout_updates["title"] = title
    fig.update_layout(layout_updates)
    return fig


def fig_climate_means(rows: pd.DataFrame, *, label: str) -> go.Figure:
    """Seasonal event-level station means with sample-size hover details."""
    fig = _base_figure()
    if rows.empty:
        fig.update_layout(title="Sem médias climáticas disponíveis para este recorte")
        return fig

    palette = {"2013": PALETTE_CATEGORICAL[0], "2014": PALETTE_CATEGORICAL[1]}
    for season in ("2013", "2014"):
        data = rows.loc[rows["season"].astype(str).eq(season)]
        if data.empty:
            continue
        customdata = data[["n", "station_name", "distance_to_circuit_km", "coverage"]].to_numpy()
        fig.add_trace(go.Scatter(
            x=data["grand_prix"], y=data["mean_c"], name=season,
            mode="markers", marker={"color": palette[season], "size": 10, "symbol": "circle"},
            customdata=customdata,
            hovertemplate=(
                f"<b>{season}</b><br>GP: %{{x}}<br>{label}: %{{y:.2f}} °C<br>"
                "Valores usados: %{customdata[0]}<br>"
                "Estação: %{customdata[1]}<br>"
                "Distância estação–circuito: %{customdata[2]:.3f} km<br>"
                "Cobertura reportada: %{customdata[3]}<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title="Grande Prêmio",
        yaxis_title=f"{label} (°C)",
        legend={"title": "Temporada"},
    )
    return fig


def fig_climate_timeline(rows: pd.DataFrame, *, value_label: str) -> go.Figure:
    """Observed-only climate points; intentionally never connects timestamps."""
    fig = _base_figure()
    if rows.empty:
        fig.update_layout(title="Sem observações numéricas para a seleção escolhida")
        return fig

    palette = {"2013": PALETTE_CATEGORICAL[0], "2014": PALETTE_CATEGORICAL[1]}
    for season in ("2013", "2014"):
        data = rows.loc[rows["season"].astype(str).eq(season)]
        if data.empty:
            continue
        customdata = data[[
            "observation_time_utc", "report_type", "quality_class_label", "quality_class",
            "quality_flag", "station_name",
        ]].to_numpy()
        fig.add_trace(go.Scatter(
            x=data["hours_from_scheduled"], y=data["measurement_c"], name=season,
            mode="markers", marker={"color": palette[season], "size": 10, "symbol": "circle"},
            customdata=customdata,
            hovertemplate=(
                f"<b>{season}</b><br>Horas desde horário programado: %{{x:.2f}} h<br>"
                f"{value_label}: %{{y:.2f}} °C<br>UTC: %{{customdata[0]}}<br>"
                "Tipo de relatório: %{customdata[1]}<br>Qualidade: %{customdata[2]}<br>"
                "ID técnico da classe: %{customdata[3]}<br>Flag: %{customdata[4]}<br>"
                "Estação: %{customdata[5]}<extra></extra>"
            ),
        ))
    fig.update_layout(
        xaxis_title="Horas desde o início programado (não é a largada real)",
        yaxis_title=f"{value_label} (°C)",
        legend={"title": "Temporada"},
    )
    return fig


def fig_historical_sensitivity(rows: pd.DataFrame) -> go.Figure:
    """Mean change from the full sample for each leave-one-GP-out scenario."""
    fig = _base_figure()
    full_rows = rows.loc[rows["gp_excluido"].eq("Nenhum")]
    data = rows.loc[rows["gp_excluido"].ne("Nenhum")].copy()
    if data.empty or full_rows.empty:
        fig.update_layout(title="Sem cenários de exclusão para exibir")
        return fig

    full_mean = float(full_rows.iloc[0]["media_seg"])
    data["impacto_seg"] = data["media_seg"] - full_mean
    data = data.sort_values("impacto_seg", key=lambda values: values.abs(), ascending=False)
    japan = data["gp_excluido"].astype(str).str.casefold().str.contains("japan|japão", regex=True)
    data["cenario_label"] = data["cenario"].astype(str).where(~japan, data["cenario"].astype(str) + " ★")
    data["direcao"] = data["impacto_seg"].map(
        lambda value: "↑ aumento" if value > 0 else "↓ redução" if value < 0 else "↔ sem variação"
    )
    colors = [ACCENT_RED if value > 0 else PALETTE_CATEGORICAL[0] for value in data["impacto_seg"]]
    customdata = data[["pares", "media_seg", "impacto_seg", "gp_excluido"]].copy()
    customdata.insert(1, "media_completa", full_mean)
    fig.add_trace(go.Bar(
        x=data["impacto_seg"], y=data["cenario_label"], orientation="h",
        text=[f"{direction} · {value:+.3f} s" for direction, value in zip(data["direcao"], data["impacto_seg"])],
        textposition="outside",
        cliponaxis=False,
        marker={
            "color": colors,
            "line": {"color": [INK_PRIMARY if highlighted else "rgba(0,0,0,0)" for highlighted in japan],
                     "width": [2 if highlighted else 0 for highlighted in japan]},
        },
        customdata=customdata.to_numpy(),
        hovertemplate=(
            "%{y}<br>Pares restantes: %{customdata[0]}<br>"
            "Média completa: %{customdata[1]:+.3f} s<br>"
            "Média sem GP: %{customdata[2]:+.3f} s<br>"
            "Variação vs completa: %{customdata[3]:+.3f} s<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line={"color": INK_MUTED, "width": 1, "dash": "dash"})
    fig.update_layout(
        xaxis_title="Variação da média vs amostra completa (s)",
        yaxis_title="GP excluído",
        yaxis={"autorange": "reversed", "automargin": True},
        xaxis={"automargin": True},
        margin={"l": 150, "r": 150, "t": 35, "b": 75},
        showlegend=False,
    )
    return fig


def fig_climate_quality(counts: pd.DataFrame) -> go.Figure:
    """Counts by TMP/DEW quality class, with labels rather than color-only state."""
    fig = _base_figure()
    if counts.empty:
        fig.update_layout(title="Sem classes de qualidade disponíveis")
        return fig
    labels = [f"{row.variable_label} · {row.quality_class_label}" for row in counts.itertuples(index=False)]
    fig.add_trace(go.Bar(
        x=counts["count"], y=labels, orientation="h", name="Observações",
        marker={"color": PALETTE_CATEGORICAL[0]},
        customdata=counts[["eligible", "quality_description", "variable", "quality_class"]].to_numpy(),
        hovertemplate=(
            "%{y}<br>Linhas: %{x}<br>Elegível QC estrito: %{customdata[0]}<br>"
            "Descrição NOAA: %{customdata[1]}<br>ID técnico da variável: %{customdata[2]}<br>"
            "ID técnico da classe: %{customdata[3]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title="Quantidade de registros",
        yaxis_title="Variável · classe QC",
        showlegend=False,
        yaxis={"autorange": "reversed"},
    )
    return fig
