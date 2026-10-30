"""Renderização das seis áreas do dashboard.

Cada função recebe apenas os dados necessários e não altera os calculadores.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analysis import compare_2013_2014, grid_probabilities
from src.dashboard.sensitivity import historical_sensitivity_summary
from src.dashboard.education import (
    SIMPLE_MODE,
    classification_explanation,
    climate_explanation,
    explorer_explanation,
    historical_explanation,
    methodology_explanation,
    overview_explanation,
    regression_explanation,
    render_explanation,
)
from src.dashboard import data as dashboard_data
from src.dashboard.data import (
    apply_explorer_filters,
    constructor_options,
    distinct_grid_options,
    driver_options,
    race_options,
    scope_summary,
)
from src.dashboard.charts import (
    fig_calibration,
    fig_delta_historical,
    fig_error_distribution,
    fig_heatmap,
    fig_lap_time_distribution,
    fig_climate_means,
    fig_climate_quality,
    fig_climate_timeline,
    fig_historical_sensitivity,
    fig_mae_comparison,
    fig_observed_vs_predicted,
    fig_precision_recall,
    fig_roc,
    fig_temporal,
    fig_grid_probabilities,
)
from src.dashboard.theme import ACCENT_RED, INK_SECONDARY, INK_MUTED, scope_note

RAW_DIR = dashboard_data.RAW_DIR


def _metric_card(label: str, value, helper: str | None = None) -> None:
    """KPI nativo com descrição auxiliar discreta (tooltip quando o Streamlit suportar)."""
    try:
        st.metric(label, value, help=helper)
    except TypeError:
        st.metric(label, value)
        if helper:
            st.markdown(f'<p class="metric-helper">{helper}</p>', unsafe_allow_html=True)


def _fmt_cell(value, na_placeholder: str) -> str:
    """Formata célula para exibição: numéricos em 'g', strings preservadas, NA como placeholder."""
    if pd.isna(value):
        return na_placeholder
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _reconcile(key: str, valid: list) -> list:
    """Mantém a seleção consistente com as opções disponíveis no recorte."""
    current = st.session_state.get(key) or []
    valid_set = set(valid)
    keep = [v for v in current if v in valid_set]
    if keep != current:
        st.session_state[key] = keep
    return keep


def _truncate(labels: list[str], limit: int = 3) -> str:
    """Lista curta de rótulos; o excedente é resumido como contagem entre parênteses."""
    if len(labels) <= limit:
        return ", ".join(labels)
    return ", ".join(labels[:limit]) + f" (+{len(labels) - limit})"


def _filter_labels(values: list | None, options: list[tuple], prefix: str = "") -> str:
    """Rótulos legíveis de uma seleção efetiva; seleção vazia significa 'todos'."""
    if not values:
        return "todos"
    selected = set(values)
    labels = [f"{prefix}{label}" for value, label in options if value in selected]
    return _truncate(labels)


def _reset_filters() -> None:
    """Limpa as seleções dos filtros exploratórios (roda antes dos widgets)."""
    for key in (
        "f1_filter_years",
        "f1_filter_races",
        "f1_filter_drivers",
        "f1_filter_constructors",
        "f1_filter_grids",
    ):
        st.session_state[key] = []


def render_overview(all_data: pd.DataFrame, reading_mode: str = SIMPLE_MODE) -> None:
    """Visão Geral: estudo completo, sem filtros analíticos."""
    st.markdown(scope_note("Escopo: base integrada padrão (2004–2017, grid > 0). Sem filtros nesta área."), unsafe_allow_html=True)

    summary = dashboard_data.scope_summary(all_data)
    years = sorted(int(y) for y in all_data["year"].unique())
    st.markdown(f"**Período efetivo:** {years[0]}–{years[-1]}")
    st.caption("Resultados disponíveis nos arquivos: 1950–2017. O calendário lista 2018, mas não há resultados dessa temporada nesta base.")
    st.divider()

    # Indicadores principais (quatro KPIs por linha em desktop; empilham em telas pequenas)
    proba = grid_probabilities(all_data)
    render_explanation(reading_mode, overview_explanation(proba, summary))
    p1 = proba[proba["grid"] == 1]
    p1_registros = int(p1["corridas"].iloc[0]) if not p1.empty else 0
    p1_vitoria = float(p1["vitoria_pct"].iloc[0]) if not p1.empty else 0.0
    p1_podio = float(p1["podio_pct"].iloc[0]) if not p1.empty else 0.0

    cols = st.columns(4)
    with cols[0]:
        _metric_card("Registros analisados", f"{summary['registros']:,}".replace(",", "."),
                     "Registros de resultado mantidos pelo pipeline (2004–2017, grid > 0).")
    with cols[1]:
        _metric_card("Corridas distintas", f"{summary['eventos']:,}".replace(",", "."),
                     "Grandes Prêmios distintos no recorte.")
    with cols[2]:
        _metric_card("Pilotos diferentes", f"{summary['pilotos']:,}".replace(",", "."),
                     "Pilotos distintos no recorte.")
    with cols[3]:
        _metric_card("Construtoras diferentes", f"{summary['construtoras']:,}".replace(",", "."),
                     "Construtoras distintas no recorte.")

    st.markdown(
        '<h4 class="spotlight-title">Destaque: largando em P1</h4>', unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    with c1:
        _metric_card("Vitória largando em P1", f"{p1_vitoria:.2f}%",
                     f"{p1_vitoria:.2f}% dos {p1_registros} registros analisados na largada em P1 "
                     "(denominador: registros analisados naquela posição de grid).")
    with c2:
        _metric_card("Pódio largando em P1", f"{p1_podio:.2f}%",
                     f"{p1_podio:.2f}% dos {p1_registros} registros analisados na largada em P1 "
                     "(denominador: registros analisados naquela posição de grid).")

    st.divider()

    # Probabilidades observadas por posição de largada
    st.subheader("Probabilidades observadas por posição de largada")
    st.caption(
        "Denominador: total de registros analisados de cada posição de grid na base padrão "
        "(2004–2017, grid > 0). Percentuais são observados no recorte histórico, não previsões."
    )
    st.plotly_chart(fig_grid_probabilities(proba), width="stretch")
    st.caption(
        "Amostras pequenas (menos de 20 registros analisados no grid) são exibidas com marcador "
        "mais discreto; os percentuais não são alterados."
    )

    with st.expander("Tabela detalhada de probabilidades por grid", expanded=False):
        display = proba.copy()
        display["grid"] = display["grid"].map(lambda g: f"P{int(g)}")
        st.dataframe(
            display.rename(columns={
                "grid": "Largada", "corridas": "Registros analisados", "vitoria_pct": "Vitória (%)",
                "podio_pct": "Pódio (%)", "top10_pct": "Top 10 (%)", "pontos_pct": "Pontuou (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Principais conclusões e limitações")
    st.markdown(
        "- Largadas da frente (P1–P3) concentram vitórias e pódios, mas a relação não é linear para todos os desfechos.\n"
        "- `grid` é a posição efetiva de largada registrada, não a colocação na qualificação.\n"
        "- `finish_position` é a ordenação do resultado (`positionOrder`), não necessariamente a chegada física ou classificação oficial.\n"
        "- Abandonos, retiradas com zero voltas e resultados sem volta rápida são preservados; ausências não invalidam a chegada/abandono.\n"
        "- Frequências são observadas no recorte histórico, não previsões garantidas.\n"
        "- A mudança de regulamento em 2014 é tratada como comparação descritiva; não se afirma causalidade.\n"
        "- Os modelos de ML são exploratórios: teste fora da amostra com baseline; não são ferramentas comprovadas para corridas atuais."
    )


def render_explorer(all_data: pd.DataFrame, reading_mode: str = SIMPLE_MODE) -> None:
    """Explorador de Corridas: único contexto com filtros exploratórios."""
    st.markdown(scope_note("Escopo: recorte filtrado por temporadas, evento, piloto, construtora e grid. Todos os indicadores desta área usam o mesmo recorte."), unsafe_allow_html=True)

    with st.sidebar:
        st.header("Filtros — Explorador")
        years_all = sorted(int(y) for y in all_data["year"].unique())
        if len(years_all) > 1:
            year_values = _reconcile("f1_filter_years", years_all)
            years = st.multiselect("Temporadas", options=years_all, key="f1_filter_years", placeholder="Todas")
        else:
            years = years_all
            st.info(f"Única temporada disponível: {years_all[0]}")

        race_opts = race_options(all_data)
        race_values = _reconcile("f1_filter_races", [r[0] for r in race_opts])
        races = st.multiselect(
            "Grande Prêmio / evento",
            options=[r[0] for r in race_opts],
            format_func=lambda rid: next(label for rid_, label in race_opts if rid_ == rid),
            key="f1_filter_races",
            placeholder="Todos",
        )

        driver_opts = driver_options(all_data)
        driver_values = _reconcile("f1_filter_drivers", [d[0] for d in driver_opts])
        drivers = st.multiselect(
            "Piloto",
            options=[d[0] for d in driver_opts],
            format_func=lambda did: next(label for did_, label in driver_opts if did_ == did),
            key="f1_filter_drivers",
            placeholder="Todos",
        )

        constructor_opts = constructor_options(all_data)
        constructor_values = _reconcile("f1_filter_constructors", [c[0] for c in constructor_opts])
        constructors = st.multiselect(
            "Construtora",
            options=[c[0] for c in constructor_opts],
            format_func=lambda cid: next(label for cid_, label in constructor_opts if cid_ == cid),
            key="f1_filter_constructors",
            placeholder="Todas",
        )

        grid_opts = distinct_grid_options(all_data)
        grid_values = _reconcile("f1_filter_grids", grid_opts)
        grids = st.multiselect("Posição de largada (grid)", options=grid_opts, key="f1_filter_grids", placeholder="Todos", max_selections=10)

        col_reset, col_reload = st.columns(2)
        with col_reset:
            st.button(
                "Redefinir filtros",
                key="f1_reset_filters",
                use_container_width=True,
                type="secondary",
                on_click=_reset_filters,
            )
        with col_reload:
            if st.button("Recarregar dados", key="f1_reload_data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

    df = dashboard_data.apply_explorer_filters(all_data, years, races, drivers, constructors, grids)
    summary = dashboard_data.scope_summary(df)
    crosstab = dashboard_data.grid_finish_crosstab(df)
    heatmap_mode = st.session_state.get("f1_heatmap_mode", "Contagem absoluta")
    render_explanation(reading_mode, explorer_explanation(summary, heatmap_mode, int(crosstab.values.sum()) if not crosstab.empty else 0))

    # Contexto dos filtros efetivamente aplicados (seleção vazia = todos).
    race_labels = [(rid, label.split(" (id=")[0]) for rid, label in race_opts]
    registros_fmt = f"{summary['registros']:,}".replace(",", ".")
    st.caption(
        "**Recorte aplicado** — "
        f"Temporadas: {_filter_labels(years, [(y, str(y)) for y in years_all])} · "
        f"Grande Prêmio: {_filter_labels(races, race_labels)} · "
        f"Pilotos: {_filter_labels(drivers, driver_opts)} · "
        f"Construtoras: {_filter_labels(constructors, constructor_opts)} · "
        f"Grid: {_filter_labels(grids, [(g, f'P{g}') for g in grid_opts])} — "
        f"{registros_fmt} registros no recorte."
    )

    st.subheader("Resumo do recorte")
    cols = st.columns(5)
    with cols[0]: _metric_card("Registros", f"{summary['registros']:,}".replace(",", "."), "Linhas de resultado no recorte")
    with cols[1]: _metric_card("Eventos distintos", f"{summary['eventos']:,}".replace(",", "."), "race_id únicos")
    with cols[2]: _metric_card("Pilotos", f"{summary['pilotos']:,}".replace(",", "."), "driver_id únicos")
    with cols[3]: _metric_card("Construtoras", f"{summary['construtoras']:,}".replace(",", "."), "constructor_id únicas")
    with cols[4]: _metric_card("Temporadas", f"{summary['temporadas']:,}".replace(",", "."), "anos distintos")

    if df.empty:
        st.markdown('<div class="empty-state">Nenhum registro corresponde aos filtros escolhidos. Revise a seleção ou use "Redefinir filtros".</div>', unsafe_allow_html=True)
        return

    st.subheader("Frequências observadas por posição de largada")
    proba = grid_probabilities(df)
    if proba.empty:
        st.info("Sem frequências calculáveis para o recorte escolhido.")
    else:
        st.plotly_chart(fig_grid_probabilities(proba), width="stretch")
        st.caption("Denominador: total de registros de cada grid neste recorte. Pontos desconhecidos tornam 'Pontuou (%)' indisponível para o grupo; nunca viram zero.")

    st.subheader("Mapa de calor: largada versus ordem final registrada")
    modo_label = st.radio(
        "Modo do mapa de calor",
        ["Contagem absoluta", "Percentual por grid"],
        key="f1_heatmap_mode",
        horizontal=True,
    )
    modo = "percent" if modo_label == "Percentual por grid" else "count"
    st.plotly_chart(fig_heatmap(crosstab, mode=modo), width="stretch")
    soma_celulas = int(crosstab.values.sum()) if not crosstab.empty else 0
    if modo == "percent":
        st.caption(
            f"Percentual condicionado ao grid: cada célula é a fração dos registros daquela posição de largada "
            f"(denominador = total da linha), não do total geral do recorte. Soma das células: {soma_celulas} registros."
        )
    else:
        st.caption(f"Soma das células: {soma_celulas} registros.")
    st.caption("Domínio derivado dos dados, incluindo posições acima de P20 quando presentes. A variação entre posição de largada e ordem final registrada não mede número de ultrapassagens.")

    with st.expander("Tabela de frequências por grid", expanded=False):
        display = proba.copy()
        display["grid"] = display["grid"].map(lambda g: f"P{int(g)}")
        st.dataframe(
            display.rename(columns={
                "grid": "Largada", "corridas": "Registros", "vitoria_pct": "Vitória (%)",
                "podio_pct": "Pódio (%)", "top10_pct": "Top 10 (%)", "pontos_pct": "Pontuou (%)",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Registros do recorte", expanded=False):
        show = df[[
            "result_id", "year", "race_id", "grand_prix", "driver", "constructor",
            "grid", "finish_position", "position", "position_text", "status", "laps", "points", "points_known",
        ]].copy()
        show["points"] = show["points"].map(lambda v: _fmt_cell(v, "—"))
        show["position"] = show["position"].map(lambda v: _fmt_cell(v, "não registrada"))
        show["laps"] = show["laps"].map(lambda v: _fmt_cell(v, "—"))
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("IDs mantêm a rastreabilidade aos arquivos originais. Campos ausentes aparecem como não registrados, nunca preenchidos por conveniência.")


def render_historical(all_data: pd.DataFrame, reading_mode: str = SIMPLE_MODE) -> None:
    """Comparação Histórica 2013–2014: contexto fixo e descritivo."""
    st.markdown(scope_note("Escopo fixo: pares de 2013 e 2014 (piloto, construtora e circuito). Filtros do Explorador não se aplicam."), unsafe_allow_html=True)

    paired, lap_stats = compare_2013_2014(all_data)
    if paired.empty:
        render_explanation(reading_mode, historical_explanation(lap_stats, pd.DataFrame()))
        st.markdown('<div class="empty-state">Não há pares comparáveis de 2013 e 2014 para os dados carregados.</div>', unsafe_allow_html=True)
        return

    sensitivity = historical_sensitivity_summary(paired)
    render_explanation(reading_mode, historical_explanation(lap_stats, sensitivity))

    # Contexto e metodologia: o que o pareamento mede e o que o sinal significa.
    st.caption(
        "Cada par reúne a volta mais rápida registrada do mesmo piloto, da mesma construtora "
        "e do mesmo circuito em 2013 e 2014. A diferença é 2014 menos 2013, em segundos: "
        "positivo significa volta mais lenta em 2014; negativo, volta mais rápida. "
        "É uma comparação descritiva — não isola pista, clima, estratégia nem regulamento."
    )
    st.warning(
        "Limitação de identidade: a Austrália 2014 apresenta inconsistências documentadas na identificação do piloto/volta mais rápida. "
        "Não há amostra reconciliada externa disponível; os pares não são homologados e não foram corrigidos automaticamente."
    )

    # Indicadores principais (valores vindos do cálculo, nunca inseridos manualmente).
    cols = st.columns(3)
    with cols[0]: _metric_card("Pares comparáveis", lap_stats["pares"], "Mesmo piloto, construtora e circuito nos dois anos")
    with cols[1]: _metric_card("Diferença média", f"{lap_stats['media_diferenca_seg']:+.3f} s", "2014 menos 2013; volta mais rápida registrada")
    with cols[2]: _metric_card("Diferença mediana", f"{lap_stats['mediana_diferenca_seg']:+.3f} s", "Mesma convenção de sinal; positivo = mais lento em 2014")

    st.subheader("Comparação por Grande Prix")
    gp_rows = []
    for name_gp, group in paired.groupby("grand_prix"):
        n = int(len(group))
        gp_rows.append({
            "grand_prix": str(name_gp),
            "diferenca_2014_menos_2013_seg": float(group["diferenca_2014_menos_2013_seg"].median()),
            "n_pares": n,
        })
    gp_rows.sort(key=lambda r: r["diferenca_2014_menos_2013_seg"])
    st.plotly_chart(fig_delta_historical(gp_rows, "diferenca_2014_menos_2013_seg", "grand_prix"), width="stretch")
    n_pequenos = sum(1 for r in gp_rows if r["n_pares"] < 5)
    if n_pequenos:
        st.caption(
            f"{n_pequenos} Grande(s) Prêmio(ões) com menos de 5 pares comparáveis exibem "
            "barra com opacidade reduzida (amostra pequena); a mediana não é ocultada."
        )

    st.subheader("Distribuição das diferenças individuais")
    st.plotly_chart(
        fig_lap_time_distribution(paired["diferenca_2014_menos_2013_seg"].to_numpy()),
        width="stretch",
    )
    st.caption(
        "Cada par conta uma vez. A soma das barras é a quantidade de pares comparáveis e "
        "valores extremos são mantidos, sem descarte arbitrário."
    )

    with st.expander("Tabela de pares comparáveis", expanded=False):
        show = paired[[
            "driver", "constructor", "circuit_id", "grand_prix",
            "volta_2013_seg", "volta_2014_seg", "diferenca_2014_menos_2013_seg",
        ]].copy()
        show["circuit_id"] = show["circuit_id"].map(lambda v: f"Circuito {int(v)}" if pd.notna(v) else "não identificado")
        show = show.rename(columns={
            "driver": "Piloto",
            "constructor": "Construtora",
            "circuit_id": "Circuito",
            "grand_prix": "Grande Prêmio",
            "volta_2013_seg": "Volta rápida 2013 (s)",
            "volta_2014_seg": "Volta rápida 2014 (s)",
            "diferenca_2014_menos_2013_seg": "Diferença (s)",
        })
        for col in ("Volta rápida 2013 (s)", "Volta rápida 2014 (s)", "Diferença (s)"):
            show[col] = show[col].map(lambda v: f"{float(v):.3f}" if pd.notna(v) else "—")
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Todos os pares válidos são preservados; a tabela não exclui valores extremos.")

    st.subheader("Sensibilidade — exclusão de um Grande Prêmio")
    st.caption(
        "A amostra original acima não é alterada. O gráfico ordena os cenários pelo impacto absoluto na média; "
        "valores positivos elevam a média e negativos a reduzem. A diferença = 2014 menos 2013."
    )
    japan = sensitivity.loc[sensitivity["cenario"].eq("Sem Japão")]
    full = sensitivity.loc[sensitivity["cenario"].eq("Amostra completa")]
    if not japan.empty and not full.empty:
        japan_row, full_row = japan.iloc[0], full.iloc[0]
        japan_mean, japan_median = _ml_number(japan_row["media_seg"]), _ml_number(japan_row["mediana_seg"])
        full_mean = _ml_number(full_row["media_seg"])
        delta = japan_mean - full_mean if japan_mean is not None and full_mean is not None else None
        cols = st.columns(3)
        with cols[0]:
            _metric_card("Pares comparáveis — sem Japão", int(japan_row["pares"]), "Cenário de sensibilidade, calculado dos pares existentes")
        with cols[1]:
            _metric_card("Diferença média — sem Japão", f"{japan_mean:+.3f} s" if japan_mean is not None else "Indisponível", "2014 menos 2013")
        with cols[2]:
            _metric_card("Diferença mediana — sem Japão", f"{japan_median:+.3f} s" if japan_median is not None else "Indisponível", "2014 menos 2013")
        if delta is None:
            st.caption("A média sem Japão não pode ser comparada porque o cenário não contém pares suficientes para calculá-la.")
        else:
            change = "aumentou" if delta > 0 else "diminuiu" if delta < 0 else "não mudou"
            st.caption(
                f"Sem Japão, a média {change} {abs(delta):.3f} s em relação à amostra completa "
                f"({delta:+.3f} s). Isso descreve sensibilidade à composição dos pares, não valida os dados."
            )
    st.plotly_chart(fig_historical_sensitivity(sensitivity), width="stretch")
    table = sensitivity.rename(columns={
        "cenario": "Cenário", "gp_excluido": "GP excluído", "pares": "Pares",
        "media_seg": "Média (s)", "mediana_seg": "Mediana (s)",
        "delta_media_vs_completa_seg": "Δ média vs completa (s)",
        "delta_mediana_vs_completa_seg": "Δ mediana vs completa (s)",
    })
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption("A tabela conserva média e mediana de todos os cenários. Esta análise não corrige identidade de voltas nem demonstra causalidade.")

    st.subheader("Conclusões e limitações")
    st.info(
        "Esta comparação é descritiva: usa a volta mais rápida registrada, pareia piloto, construtora e circuito, e não isola "
        "efeitos de pista, clima, estratégia ou regulamento. Não há identificação causal da mudança de regulamento de 2014."
    )


def _climate_bool(value) -> bool:
    return value is True or str(value).strip().casefold() in {"true", "1", "yes"}


def _quality_class_label(value: str) -> str:
    return {
        "passed_all_qc": "Qualidade aprovada",
        "suspect": "Valor suspeito",
    }.get(str(value), str(value))


def render_climate(reading_mode: str = SIMPLE_MODE) -> None:
    """Local 2013–2014 station observations; never paired with lap times."""
    st.markdown(scope_note("Observações NOAA em estações regionais e janelas programadas; não são medições da pista."), unsafe_allow_html=True)
    try:
        bundle = dashboard_data.load_climate_bundle(dashboard_data.climate_file_signature())
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Não foi possível carregar os dados climáticos locais: {exc}")
        return

    obs, events = bundle["observations"].copy(), bundle["events"]
    summary = bundle["event_summary"].copy()
    if obs.empty or events.empty:
        st.info("Os arquivos locais existem, mas não há eventos ou observações disponíveis.")
        render_explanation(reading_mode, climate_explanation(obs, events))
        return
    tmp = pd.to_numeric(obs["air_temperature_c"], errors="coerce")
    dew = pd.to_numeric(obs["dew_point_c"], errors="coerce")
    tmp_ok = obs["tmp_qc_eligible"].map(_climate_bool) & tmp.notna()
    dew_ok = obs["dew_qc_eligible"].map(_climate_bool) & dew.notna()
    covered = int(summary.loc[summary["observation_count"].gt(0), "race_id"].nunique())
    render_explanation(reading_mode, climate_explanation(obs, events))
    st.info(
        "A janela usa o início programado, não a largada real. Não há vínculo temporal entre clima e voltas rápidas; "
        "não são calculadas correlação, regressão ou causalidade."
    )
    cols = st.columns(3)
    with cols[0]: _metric_card("GPs investigados", int(events["grand_prix"].nunique()), "Cinco GPs em 2013–2014")
    with cols[1]: _metric_card("Eventos com registros", f"{covered}/{len(events)}", "Presença de observações; não indica continuidade")
    with cols[2]: _metric_card("Observações", len(obs), "Linhas nas janelas locais")
    cols = st.columns(2)
    with cols[0]: _metric_card("Temperatura do ar QC-aceita", f"{int(tmp_ok.sum())}/{int(tmp.notna().sum())}", "Flags 1 e 5; elegibilidade independente do ponto de orvalho")
    with cols[1]: _metric_card("Ponto de orvalho QC-aceito", f"{int(dew_ok.sum())}/{int(dew.notna().sum())}", "Flags 1 e 5; elegibilidade independente da temperatura do ar")

    gp_tab, timeline_tab, quality_tab = st.tabs(["GP e cobertura", "Evolução temporal", "Qualidade e evidências"])
    with gp_tab:
        policy = st.radio("Valores para as médias", ["Todos os valores numéricos", "Somente QC-aceitos"], horizontal=True, key="climate_mean_policy")
        qc = policy == "Somente QC-aceitos"
        tmp_numeric_count = int(tmp.notna().sum())
        tmp_accepted_count = int(tmp_ok.sum())
        if tmp_accepted_count == tmp_numeric_count and tmp_numeric_count:
            st.info(
                f"As médias de temperatura do ar não mudam ao selecionar somente valores aprovados: "
                f"{tmp_accepted_count}/{tmp_numeric_count} observações numéricas disponíveis passaram os critérios QC. "
                "A aprovação da estação não comprova as condições meteorológicas no circuito."
            )
        else:
            st.caption(
                f"Valores aceitos no QC estrito: {tmp_accepted_count} de {tmp_numeric_count} observações numéricas; "
                "a aprovação da estação não comprova as condições meteorológicas no circuito."
            )
        value_col = "tmp_mean_qc_accepted_c" if qc else "tmp_mean_all_c"
        count_col = "tmp_qc_accepted_count" if qc else "tmp_numeric_count_all"
        station_by_race = obs.groupby("race_id")["station_name"].first().to_dict()
        rows = []
        for row in summary.to_dict("records"):
            mean = _ml_number(row.get(value_col))
            if mean is not None:
                rows.append({
                    "season": str(row["season"]), "grand_prix": row["grand_prix"], "mean_c": mean,
                    "n": int(row.get(count_col, 0)), "station_name": station_by_race.get(row["race_id"], ""),
                    "distance_to_circuit_km": row["distance_to_circuit_km"],
                    "coverage": f"{row['unique_timestamp_count']} timestamps; {row['report_type_counts']}",
                })
        st.subheader("Temperatura do ar por Grande Prêmio")
        st.plotly_chart(fig_climate_means(pd.DataFrame(rows), label="Média QC-aceita" if qc else "Média de todos os valores"), width="stretch")
        st.caption(
            "Média aritmética por observação, não ponderada pelo tempo. O hover do gráfico informa n, estação/distância "
            "ao circuito e cobertura temporal; estação aprovada não comprova as condições meteorológicas no circuito."
        )
        names = {
            "season": "Temporada", "grand_prix": "Grande Prêmio", "observation_count": "Linhas",
            "unique_timestamp_count": "Timestamps únicos", "first_observation_utc": "Primeiro UTC",
            "last_observation_utc": "Último UTC", "report_type_counts": "Tipos de relatório",
            "distance_to_circuit_km": "Distância estação–circuito (km)",
            "tmp_numeric_count_all": "Temperatura do ar numérica (n)", "tmp_mean_all_c": "Temperatura média total (°C)",
            "tmp_qc_accepted_count": "Temperatura QC aprovada (n)", "tmp_mean_qc_accepted_c": "Temperatura média QC (°C)",
            "dew_numeric_count_all": "Ponto de orvalho numérico (n)", "dew_mean_all_c": "Ponto de orvalho médio total (°C)",
            "dew_qc_accepted_count": "Ponto de orvalho QC aprovado (n)", "dew_mean_qc_accepted_c": "Ponto de orvalho médio QC (°C)",
        }
        with st.expander("Cobertura e estatísticas por evento", expanded=False):
            st.dataframe(summary.rename(columns=names), width="stretch", hide_index=True)
        with st.expander("Tipos de relatório comuns", expanded=False):
            st.caption("Tipos comuns não significam timestamps pareados nem cobertura equivalente.")
            st.dataframe(bundle["common_types"], width="stretch", hide_index=True)

    with timeline_tab:
        gp_options = sorted(obs["grand_prix"].dropna().astype(str).unique())
        gp = st.selectbox("Grande Prêmio", gp_options, key="climate_timeline_gp")
        variable = st.selectbox("Variável", ["Temperatura do ar", "Ponto de orvalho"], key="climate_timeline_var")
        policy = st.selectbox("Valores", ["Todos os valores numéricos", "Somente QC-aceitos"], key="climate_timeline_policy")
        prefix, raw_col, label = ("tmp", "air_temperature_c", "Temperatura do ar") if variable.startswith("Temperatura") else ("dew", "dew_point_c", "Ponto de orvalho")
        frame = obs.loc[obs["grand_prix"].astype(str).eq(gp)].copy()
        frame["measurement_c"] = pd.to_numeric(frame[raw_col], errors="coerce")
        timestamp = pd.to_datetime(frame["observation_time_utc"], utc=True, errors="coerce")
        scheduled = pd.to_datetime(frame["scheduled_start_utc"], utc=True, errors="coerce")
        bad_time = int((timestamp.isna() | scheduled.isna()).sum())
        frame["hours_from_scheduled"] = (timestamp - scheduled).dt.total_seconds() / 3600
        frame["quality_class"] = frame[f"{prefix}_quality_class"].fillna("missing_class").astype(str)
        frame["quality_class_label"] = frame["quality_class"].map(_quality_class_label)
        frame["quality_flag"] = frame[f"{prefix}_quality_flag"].fillna("").astype(str)
        frame["eligible"] = frame[f"{prefix}_qc_eligible"].map(_climate_bool)
        frame["station_name"] = frame["station_name"].fillna("").astype(str)
        frame = frame.loc[frame["measurement_c"].notna() & timestamp.notna() & scheduled.notna()]
        if policy == "Somente QC-aceitos":
            frame = frame.loc[frame["eligible"]]
        if bad_time:
            st.warning(f"{bad_time} linha(s) com horário inválido foram excluídas do gráfico.")
        st.plotly_chart(fig_climate_timeline(frame, value_label=label), width="stretch")
        st.caption("Horas desde o início programado (não é a largada real). Pontos observados sem conexão ou interpolação; lacunas permanecem sem marcas.")
        with st.expander("Tabela de observações selecionadas", expanded=False):
            st.dataframe(frame[["season", "observation_time_utc", "hours_from_scheduled", "report_type", "measurement_c", "quality_flag", "quality_class_label", "quality_class", "eligible", "station_name"]].rename(columns={
                "quality_class_label": "Classe de qualidade", "quality_class": "ID técnico da classe QC",
            }), width="stretch", hide_index=True)

    with quality_tab:
        records = []
        for prefix, col, label in (
            ("tmp", "air_temperature_c", "Temperatura do ar"),
            ("dew", "dew_point_c", "Ponto de orvalho"),
        ):
            numeric = pd.to_numeric(obs[col], errors="coerce").notna()
            subset = obs.loc[numeric].copy()
            subset["eligible"] = subset[f"{prefix}_qc_eligible"].map(_climate_bool)
            subset["quality_class"] = subset[f"{prefix}_quality_class"].fillna("missing_class").astype(str)
            subset["quality_class_label"] = subset["quality_class"].map(_quality_class_label)
            subset["quality_description"] = subset[f"{prefix}_quality_description"].fillna("").astype(str)
            grouped = subset.groupby(
                ["quality_class", "quality_class_label", "quality_description", "eligible"], dropna=False,
            ).size().reset_index(name="count")
            for row in grouped.to_dict("records"):
                records.append({"variable": prefix.upper(), "variable_label": label, **row})
        quality = pd.DataFrame(records)
        st.subheader("Qualidade das observações por variável")
        st.plotly_chart(fig_climate_quality(quality), width="stretch")
        quality_table = quality.rename(columns={
            "variable_label": "Variável", "variable": "ID da variável",
            "quality_class_label": "Classe de qualidade", "quality_class": "ID da classe QC",
            "quality_description": "Descrição NOAA", "eligible": "Elegível QC estrito",
            "count": "Registros numéricos",
        })
        st.dataframe(quality_table[[
            "Variável", "ID da variável", "Classe de qualidade", "ID da classe QC",
            "Descrição NOAA", "Elegível QC estrito", "Registros numéricos",
        ]], width="stretch", hide_index=True)
        st.caption("Política estrita: flags 1 e 5 (“passou todos os controles”). Temperatura do ar e ponto de orvalho são avaliados separadamente.")
        st.markdown("Documentação: [NOAA/NCEI ISD Format Document, p. 11](https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf).")
        st.info("QC aprovado é uma propriedade da observação na estação, não sua representatividade da pista. Sensibilidade histórica não valida os dados nem testa causalidade.")
        with st.expander("Evidências históricas documentadas (não são medições)", expanded=False):
            evidence = bundle["evidence"].rename(columns={
                "season": "Temporada", "grand_prix": "Grande Prêmio", "session": "Sessão",
                "fact_type": "Tipo de evidência", "description": "Relato documentado", "source_class": "Classe da fonte",
                "source_url": "URL da fonte", "confidence": "Confiança", "limitation": "Limitação",
            })
            st.dataframe(evidence[["Temporada", "Grande Prêmio", "Sessão", "Tipo de evidência", "Relato documentado", "Classe da fonte", "Confiança", "Limitação", "URL da fonte"]], width="stretch", hide_index=True)
            st.caption("Relatos de classificação, pré-evento e corrida permanecem identificados pela sessão; narrativa não é observação instrumental.")


def _ml_number(value) -> float | None:
    """Converte apenas métricas escalares finitas para apresentação."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ml_value(value, digits: int = 3) -> str:
    number = _ml_number(value)
    return f"{number:.{digits}f}" if number is not None else "Indisponível"


def _regression_interpretation(reg: dict, refs: dict) -> str:
    """Compara apenas métricas reportadas, sem refazer estimativas."""
    parts = ["MAE é o erro absoluto médio em posições; quanto menor, melhor."]
    comparisons = (
        ("mae", "mae_previsao_posicao_igual_ao_grid", "MAE", "menor"),
        ("rmse", "rmse_previsao_posicao_igual_ao_grid", "RMSE", "menor"),
        ("r2", "r2_previsao_posicao_igual_ao_grid", "R²", "maior"),
    )
    for model_key, ref_key, label, best in comparisons:
        model_value, ref_value = _ml_number(reg.get(model_key)), _ml_number(refs.get(ref_key))
        if model_value is None or ref_value is None:
            parts.append(f"Comparação de {label} com a referência grid: indisponível.")
            continue
        if model_value == ref_value:
            parts.append(f"Modelo e previsão = grid empatam em {label} ({_ml_value(model_value)}).")
        else:
            better = (model_value < ref_value) if best == "menor" else (model_value > ref_value)
            result = "supera" if better else "não supera"
            parts.append(
                f"Em {label}, a regressão {result} a previsão = grid "
                f"({_ml_value(model_value)} contra {_ml_value(ref_value)})."
            )
    parts.append("O conjunto de métricas não autoriza afirmar superioridade geral do modelo.")
    return " ".join(parts)


def _classification_interpretation(clf: dict) -> str:
    """Explica métricas e trade-offs disponíveis sem recomputar o classificador."""
    if clf.get("erro_classificador"):
        return "Classificador indisponível neste recorte; não há desempenho de classificação a interpretar."
    parts = [
        "Precisão é a proporção de pódios entre previsões de pódio; recall é a proporção de pódios reais identificados. "
        "F1 resume precisão e recall. ROC-AUC mede separação ao variar o limiar; Average Precision (AP) resume "
        "a curva precisão–recall. Brier Score e Log Loss avaliam probabilidades (menor é melhor)."
    ]
    recall, precision = _ml_number(clf.get("recall")), _ml_number(clf.get("precision"))
    if recall is not None and precision is not None:
        if recall >= 0.8 and precision < 0.7:
            parts.append(
                f"No limiar adotado, o recall é alto ({_ml_value(recall)}), mas a precisão é moderada "
                f"({_ml_value(precision)}): mais pódios capturados, com falsos positivos."
            )
        else:
            parts.append(
                f"No limiar adotado, recall = {_ml_value(recall)} e precisão = {_ml_value(precision)}; "
                "a relação entre acertos e falsos alertas depende destes valores."
            )
    else:
        parts.append("Precisão e/ou recall indisponíveis; não se infere o trade-off do limiar.")
    auc, grid_auc = _ml_number(clf.get("roc_auc")), _ml_number(clf.get("grid_ranking_auc"))
    if auc is not None and grid_auc is not None and auc == grid_auc:
        parts.append("O ROC-AUC coincide, nos valores reportados, com o ranking simples por posição de grid.")
    elif auc is not None and grid_auc is not None:
        parts.append("O ROC-AUC difere do ranking simples por posição de grid.")
    parts.append(
        "O limiar foi selecionado no treino; a escolha intra-amostral pode ser otimista. "
        "Teste temporal e baselines não comprovam utilidade em corridas atuais ou apostas."
    )
    return " ".join(parts)


def _ml_table(rows) -> pd.DataFrame:
    """Renomeia colunas e serializa campos aninhados para tabelas Arrow."""
    frame = pd.DataFrame(rows).rename(columns={
        "temporada": "Temporada", "ano_validacao": "Temporada", "ano": "Temporada",
        "n": "Amostras", "mae": "MAE", "rmse": "RMSE", "r2": "R²",
        "amostras_treino": "Amostras de treino", "amostras_validacao": "Amostras de validação",
    })
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            )
    return frame


def render_ml(all_data: pd.DataFrame, signature: str, reading_mode: str = SIMPLE_MODE) -> None:
    """Laboratório de Machine Learning: experimento fixo, cacheado."""
    st.markdown(scope_note("Escopo fixo: treino 2004–2014, teste final 2015–2017. Filtros do Explorador não se aplicam."), unsafe_allow_html=True)

    try:
        stats, arrays = dashboard_data.evaluate_model_experiment(all_data, signature)
    except ValueError as exc:
        st.error(f"Não foi possível executar a avaliação de ML: {exc}")
        return

    st.subheader("Regressão linear — ordenação final")
    reg = stats["regressao_linear"]
    refs = stats["referencias"]
    render_explanation(reading_mode, regression_explanation(reg, refs))
    cols = st.columns(3)
    with cols[0]: _metric_card("MAE: regressão linear", _ml_value(reg.get("mae")), "posições; menor é melhor")
    with cols[1]: _metric_card("MAE: previsão = grid", _ml_value(refs.get("mae_previsao_posicao_igual_ao_grid")), "referência simples")
    with cols[2]: _metric_card("MAE: mediana do treino", _ml_value(refs.get("mae_previsao_constante_mediana")), "referência constante")

    detail_cols = st.columns(4)
    with detail_cols[0]: _metric_card("RMSE: modelo", _ml_value(reg.get("rmse")), "posições")
    with detail_cols[1]: _metric_card("RMSE: grid", _ml_value(refs.get("rmse_previsao_posicao_igual_ao_grid")), "posições")
    with detail_cols[2]: _metric_card("R²: modelo", _ml_value(reg.get("r2")), "coeficiente de determinação")
    with detail_cols[3]: _metric_card("R²: grid", _ml_value(refs.get("r2_previsao_posicao_igual_ao_grid")), "referência")

    st.subheader("MAE: modelo versus referências")
    mae_values = {
        label: value for label, value in (
            ("Modelo: regressão linear", reg.get("mae")),
            ("Referência: previsão = grid", refs.get("mae_previsao_posicao_igual_ao_grid")),
            ("Referência: mediana do treino", refs.get("mae_previsao_constante_mediana")),
        ) if _ml_number(value) is not None
    }
    if mae_values:
        st.plotly_chart(fig_mae_comparison(mae_values, title=None), width="stretch")
    else:
        st.info("Comparação de MAE indisponível.")
    st.info(_regression_interpretation(reg, refs))

    with st.expander("Diagnósticos da regressão: previsões e resíduos", expanded=False):
        observed, predicted = arrays.get("finish_position_test"), arrays.get("regression_predictions_test")
        st.subheader("Observado versus previsto (regressão linear)")
        if observed is not None and predicted is not None and len(observed) and len(predicted):
            st.plotly_chart(fig_observed_vs_predicted(observed, predicted, title=None), width="stretch")
        else:
            st.info("Pares observado/previsto indisponíveis.")
        st.subheader("Distribuição dos erros (observado − previsto)")
        errors = arrays.get("regression_errors")
        if errors is not None and len(errors):
            st.plotly_chart(fig_error_distribution(errors, title=None), width="stretch")
        else:
            st.info("Resíduos indisponíveis.")
        st.caption("Resíduo = ordenação observada menos previsão contínua, em posições.")
        err_cols = st.columns(5)
        with err_cols[0]: _metric_card("Erro médio", _ml_value(reg.get("mean_error")), "posições; observado − previsto")
        with err_cols[1]: _metric_card("Erro mediano", _ml_value(reg.get("median_error")), "posições")
        with err_cols[2]: _metric_card("Desvio padrão", _ml_value(reg.get("std_error")), "posições")
        with err_cols[3]: _metric_card("Fora do intervalo válido", reg.get("outside_valid_range_count") if reg.get("outside_valid_range_count") is not None else "Indisponível", "contagem no teste")
        with err_cols[4]: _metric_card("Intervalo válido", f"[{reg['valid_range_min']}, {reg['valid_range_max']}]" if reg.get("valid_range_min") is not None and reg.get("valid_range_max") is not None else "Indisponível", "posições")

    with st.expander("Temporalidade — regressão", expanded=False):
        per_season = reg.get("per_season_teste") or {}
        st.subheader("Métricas por temporada de teste — regressão")
        if per_season:
            st.dataframe(_ml_table([{"temporada": year, **metrics} for year, metrics in per_season.items()]), width="stretch", hide_index=True)
        else:
            st.info("Sem métricas por temporada disponíveis.")
        st.subheader("Validação progressiva dentro do treino — regressão")
        temp_reg = stats.get("validacao_temporal_treino", {}).get("regressao") or []
        if temp_reg:
            st.plotly_chart(fig_temporal(temp_reg, "mae", "MAE por janela de validação — regressão"), width="stretch")
            st.dataframe(_ml_table(temp_reg), width="stretch", hide_index=True)
        else:
            st.info("Sem janelas de validação progressiva disponíveis.")

    st.divider()
    st.subheader("Classificação de pódio (posição registrada ≤ 3)")
    clf = stats["classificacao_podio"]
    render_explanation(reading_mode, classification_explanation(clf))
    bins = clf.get("calibration_bins") or []
    ccols = st.columns(4)
    with ccols[0]: _metric_card("ROC AUC", _ml_value(clf.get("roc_auc")), "separação ao variar o limiar")
    with ccols[1]: _metric_card("Average Precision (AP)", _ml_value(clf.get("pr_auc")), "campo pr_auc; não é integral trapezoidal")
    with ccols[2]: _metric_card("Brier score", _ml_value(clf.get("brier_score")), "menor é melhor")
    with ccols[3]: _metric_card("Log loss", _ml_value(clf.get("log_loss")), "menor é melhor")

    pcols = st.columns(4)
    with pcols[0]: _metric_card("Precisão", _ml_value(clf.get("precision")), "no limiar escolhido no treino")
    with pcols[1]: _metric_card("Recall", _ml_value(clf.get("recall")), "no limiar escolhido no treino")
    with pcols[2]: _metric_card("F1", _ml_value(clf.get("f1")), "no limiar escolhido no treino")
    with pcols[3]: _metric_card("Limiar (treino)", _ml_value(clf.get("threshold")), "critério de Youden, apenas treino")

    cal_cols = st.columns(3)
    with cal_cols[0]: _metric_card("ECE", _ml_value(clf.get("calibration_ece")), "erro esperado de calibração")
    with cal_cols[1]: _metric_card("MCE", _ml_value(clf.get("calibration_mce")), "erro máximo entre bins ocupados")
    with cal_cols[2]: _metric_card("Bins ocupados", sum(1 for b in bins if b.get("count", 0) > 0) if bins else "Indisponível", "bins vazios não viram frequência zero")

    st.subheader("Matriz de confusão (observado × previsto)")
    cm = clf.get("confusion_matrix")
    if cm is not None and len(cm) == 2 and all(len(row) == 2 for row in cm):
        cm_df = pd.DataFrame(cm, index=["Observado: não pódio", "Observado: pódio"], columns=["Previsto: não pódio", "Previsto: pódio"])
        st.dataframe(cm_df, width="stretch")
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        st.caption(
            f"Acertos: verdadeiros negativos (não pódios) {tn}; verdadeiros positivos (pódios) {tp}. "
            f"Erros: falsos positivos (pódio previsto, não ocorreu) {fp}; falsos negativos "
            f"(pódio não previsto) {fn}. Suporte: {tn + fp} não pódios e {fn + tp} pódios; "
            f"total {tn + fp + fn + tp} registros. Linhas = observado; colunas = previsto."
        )
    else:
        st.info("Matriz de confusão indisponível (classificador não ajustável ou classe insuficiente).")

    proba_test = arrays.get("classifier_probabilities_test")
    y_bin = arrays.get("target_test")
    can_plot_curves = (
        proba_test is not None and y_bin is not None
        and len(proba_test) == len(y_bin) and len(y_bin)
        and len(set(y_bin.tolist())) == 2
    )
    st.subheader("Curva ROC")
    if can_plot_curves:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_bin, proba_test)
        st.plotly_chart(fig_roc(fpr, tpr, clf.get("roc_auc"), title=None), width="stretch")
    else:
        st.info("Curva ROC indisponível (classificador ou classes insuficientes no teste).")

    st.subheader("Curva precisão–recall")
    if can_plot_curves:
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_bin, proba_test)
        st.plotly_chart(fig_precision_recall(recall, precision, clf.get("pr_auc"), clf.get("train_prevalence"), title=None), width="stretch")
        st.caption("Average Precision (AP) é o valor já computado (`pr_auc`); não se deriva da integração trapezoidal desta curva.")
    else:
        st.info("Curva precisão–recall indisponível (classificador ou classes insuficientes no teste).")

    st.subheader("Curva de calibração")
    if bins:
        st.plotly_chart(fig_calibration(bins, title=None), width="stretch")
        st.caption("O hover mostra os registros de cada faixa ocupada; faixas vazias não recebem frequência observada.")
        with st.expander("Registros por faixa de probabilidade", expanded=False):
            bin_rows = [{
                "Faixa": b.get("bin"), "Probabilidade média": b.get("mean_predicted"),
                "Registros": b.get("count", 0),
                "Fração observada": b.get("fraction_positives") if b.get("count", 0) and pd.notna(b.get("fraction_positives")) else None,
            } for b in bins]
            st.dataframe(pd.DataFrame(bin_rows), width="stretch", hide_index=True)
            st.caption("Células vazias na fração observada indicam faixas sem registros; não representam zero.")
    else:
        st.info("Curva de calibração indisponível.")

    st.info(_classification_interpretation(clf))

    with st.expander("Temporalidade — classificação", expanded=False):
        st.subheader("Métricas por temporada de teste — classificação")
        per_season_clf = clf.get("per_season_teste") or {}
        if per_season_clf:
            st.dataframe(_ml_table([{"temporada": year, **metrics} for year, metrics in per_season_clf.items()]), width="stretch", hide_index=True)
        else:
            st.info("Sem métricas por temporada disponíveis.")
        st.subheader("Validação progressiva dentro do treino — classificação")
        temp_clf = stats.get("validacao_temporal_treino", {}).get("classificacao") or []
        if temp_clf:
            st.dataframe(_ml_table(temp_clf), width="stretch", hide_index=True)
        else:
            st.info("Sem janelas de validação progressiva disponíveis.")


def render_data_methodology(all_data: pd.DataFrame, signature: str, reading_mode: str = SIMPLE_MODE) -> None:
    """Dados e Metodologia: transparência sobre a fonte e a avaliação."""
    st.markdown(scope_note("Escopo: auditoria dos arquivos atuais e contrato da base integrada. Independente de filtros do Explorador."), unsafe_allow_html=True)

    st.subheader("Origem e cobertura dos dados")
    st.markdown(
        "- Cinco CSVs contratados: `results`, `races`, `drivers`, `constructors`, `status`.\n"
        "- Resultados: 1950–2017. Calendário: até 2018; os 21 eventos de 2018 não têm resultados nesta base.\n"
        "- Recorte padrão da análise: 2004–2017, grid > 0.\n"
        "- `grid` é a posição efetiva de largada registrada; não é colocação na qualificação.\n"
        "- `finish_position` vem de `positionOrder` (ordenação do resultado), não da classificação oficial.\n"
    )

    st.subheader("Auditoria completa dos arquivos (calculada em memória)")
    try:
        report = dashboard_data.load_audit_report(str(RAW_DIR), signature)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Não foi possível auditar os arquivos: {exc}")
        report = None

    render_explanation(reading_mode, methodology_explanation(report))
    if report:
        val = report.get("validation", {})
        mcols = st.columns(3)
        with mcols[0]: _metric_card("Achados bloqueantes", val.get("blocking_issues", 0), "bloqueiam o pipeline no modo padrão")
        with mcols[1]: _metric_card("Avisos (warnings)", val.get("warning_issues", 0), "não bloqueiam o padrão; contam do relatório atual")
        with mcols[2]: _metric_card("Tabelas auditadas", len(report.get("tables", {})), "results, races, drivers, constructors, status")

        issues = report.get("issues", [])
        if issues:
            st.subheader("Achados por severidade")
            sev_order = {"error": 0, "warning": 1, "info": 2}
            ordered = sorted(issues, key=lambda i: (sev_order.get(i.get("severity", "info"), 3), i.get("table", ""), i.get("code", "")))
            rows = [{
                "Tabela": it.get("table"),
                "Código": it.get("code"),
                "Severidade": it.get("severity"),
                "Contagem": it.get("count"),
                "Bloqueante (padrão)": "Sim" if it.get("blocking") else "Não",
                "Mensagem": it.get("message"),
            } for it in ordered]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Contagem de achados ≠ quantidade de registros únicos afetados. Exemplos são amostras limitadas por regra.")
        else:
            st.success("Nenhum achado de auditoria nos arquivos atuais.")

        prov = report.get("provenance", {})
        st.subheader("Proveniência e direitos de uso")
        st.markdown(
            f"- Distribuição identificada: {prov.get('identified_distribution') or 'PENDENTE'} "
            f"(publicador: {prov.get('distribution_publisher') or 'PENDENTE'})\n"
            f"- URL da distribuição: {prov.get('distribution_url') or 'PENDENTE'}\n"
            f"- Fonte primária anterior ao Kaggle: {prov.get('source') or prov.get('upstream_source_status') or 'PENDENTE'}\n"
            f"- URL da fonte primária: {prov.get('source_url') or 'PENDENTE'}\n"
            f"- Versão/release exato: {prov.get('dataset_version') or 'PENDENTE'}\n"
            f"- Data/método de extração: {prov.get('extracted_at') or 'PENDENTE'}\n"
            f"- Licença aplicável à cadeia de dados: {prov.get('license') or 'PENDENTE'}\n"
            f"- Licença declarada na página do Kaggle: {prov.get('declared_license') or 'PENDENTE'}\n"
            f"- Correspondência dos cinco CSVs: {prov.get('local_file_match_evidence') or 'PENDENTE'}\n"
            f"- Permissão de redistribuição pública: {prov.get('redistribution_permission') or 'PENDENTE; não confirmada'}\n"
        )
        st.warning(
            "A distribuição, a licença declarada e a correspondência dos arquivos não identificam a fonte primária anterior "
            "nem confirmam os direitos de todos os componentes. A redistribuição pública permanece não confirmada; "
            "avalie as condições NC/SA e os direitos aplicáveis antes de publicar CSVs ou derivados."
        )

    st.subheader("Contrato de ingestão e critérios de inclusão/exclusão")
    st.markdown(
        "- Junções muitos-para-um validadas; PKs/FKs inválidas bloqueiam antes de qualquer junção.\n"
        "- Nenhuma multiplicação ou perda silenciosa de resultados.\n"
        "- Registros com grid = 0 são excluídos com contagem explícita (informativa, não abandono).\n"
        "- Abandonos, DSQs, retiradas com zero voltas e resultados sem volta rápida são preservados.\n"
        "- Não se aplica filtro `laps > 0` e não se deduplicam grids arbitrariamente.\n"
        "- Pontos ausentes permanecem `NaN`; se houver qualquer ponto desconhecido em um grid, `pontos_pct` fica indisponível para esse grupo.\n"
    )

    st.subheader("Limitações estatísticas e de Machine Learning")
    st.markdown(
        "- Frequências observadas não são probabilidades universais nem chances futuras garantidas.\n"
        "- A comparação 2013–2014 é descritiva; não demonstra causalidade do regulamento.\n"
        "- O ML usa apenas `grid` como entrada, com divisão temporal e baselines; teste fora do período de treino.\n"
        "- A regressão linear não superou o baseline de grid em MAE; o AUC do classificador iguala o ranking simples por grid.\n"
        "- Modelos não são apresentados como ferramentas comprovadas de previsão de corridas atuais.\n"
    )

    dq = all_data.attrs.get("data_quality") or {}
    if dq:
        st.subheader("Relatório da base integrada carregada")
        st.json(dq, expanded=False)
