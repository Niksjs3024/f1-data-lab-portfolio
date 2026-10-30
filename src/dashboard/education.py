"""Reusable plain-language explanations and glossary for the dashboard."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import streamlit as st

SIMPLE_MODE = "Explicação simples"
TECHNICAL_MODE = "Visão técnica"
READING_MODES = (SIMPLE_MODE, TECHNICAL_MODE)

GLOSSARY = (
    ("Dataset", "Conjunto organizado de dados usado em uma análise."),
    ("Amostra", "Parte dos dados que atende aos critérios de uma análise; seus resultados descrevem essa parte."),
    ("Média", "Soma dos valores dividida pela quantidade de valores; valores extremos podem puxá-la."),
    ("Mediana", "Valor central após ordenar os dados; costuma ser menos afetada por extremos."),
    ("Posição de largada (grid)", "Lugar registrado para a largada; não é necessariamente a posição da classificação."),
    ("Pódio", "Neste estudo, posição final registrada entre as três primeiras."),
    ("Baseline", "Regra simples de referência para avaliar se um modelo acrescenta informação."),
    ("Machine Learning", "Métodos que ajustam padrões nos dados para produzir estimativas; não garantem previsões corretas."),
    ("MAE", "Erro absoluto médio: distância média entre previsão e resultado, aqui medida em posições."),
    ("Precisão", "Entre os casos que o modelo marcou como pódio, a fração que realmente foi pódio."),
    ("Recall", "Entre os pódios reais, a fração que o modelo conseguiu identificar."),
    ("ROC-AUC", "Mede a capacidade de ordenar positivos acima de negativos ao variar o limiar; não mede calibração."),
    ("Average Precision", "Resumo da curva precisão–recall, especialmente informativo quando as classes têm tamanhos diferentes."),
    ("Qualidade NOAA", "Flags de controle da observação meteorológica; aprovação não prova que a estação represente a pista."),
    ("Análise de sensibilidade", "Verifica como uma descrição muda ao alterar uma escolha, como excluir um GP, sem validar os dados."),
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_pt_br(value: Any, digits: int = 2, suffix: str = "") -> str:
    """Format finite numeric values with Brazilian decimal/group separators."""
    number = _finite(value)
    if number is None:
        return "indisponível"
    formatted = f"{number:,.{digits}f}"
    formatted = formatted.replace(",", "\0").replace(".", ",").replace("\0", ".")
    return f"{formatted}{suffix}"


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    return format_pt_br(value, digits, suffix)


def _fmt_signed(value: Any, digits: int = 3, suffix: str = "") -> str:
    number = _finite(value)
    if number is None:
        return "indisponível"
    return format_pt_br(number, digits, suffix) if number < 0 else f"+{format_pt_br(number, digits, suffix)}"


def _fmt_percent(value: Any, digits: int = 1) -> str:
    number = _finite(value)
    return format_pt_br(number * 100, digits, "%") if number is not None else "indisponível"


def _fmt_count(value: Any) -> str:
    number = _finite(value)
    return f"{int(number):,}".replace(",", ".") if number is not None else "indisponível"


def _result_direction(value: float | None, *, positive: str, negative: str, neutral: str) -> str:
    if value is None:
        return "O resultado não está disponível neste recorte."
    if value > 0:
        return positive
    if value < 0:
        return negative
    return neutral


def overview_explanation(proba: pd.DataFrame, summary: Mapping[str, Any]) -> tuple[str, str, str]:
    p1 = proba.loc[proba["grid"].eq(1)] if "grid" in proba else pd.DataFrame()
    if p1.empty:
        finding = "Não há frequência calculável para a posição de largada P1 neste recorte."
    else:
        row = p1.iloc[0]
        n = int(row.get("corridas", 0))
        win = _finite(row.get("vitoria_pct"))
        podium = _finite(row.get("podio_pct"))
        metrics = []
        if win is not None:
            metrics.append(f"{_fmt(win, 1, '%')} venceram")
        if podium is not None:
            metrics.append(f"{_fmt(podium, 1, '%')} chegaram ao pódio")
        finding = (
            f"Entre {_fmt_count(n)} largadas em P1, " + (" e ".join(metrics) if metrics else "as frequências estão indisponíveis") + "."
        )
    records, events = summary.get("registros"), summary.get("eventos")
    unit_note = (
        f"A base reúne {_fmt_count(records)} resultados em {_fmt_count(events)} corridas."
        if records is not None and events is not None else
        "Cada resultado representa um piloto em uma corrida."
    )
    return (
        "Comparamos a posição de largada com vitória e pódio. P1 é a primeira posição registrada no grid.",
        finding + " " + unit_note,
        "São frequências históricas observadas, não previsões para uma próxima corrida. Grupos pequenos podem variar mais."
    )


def explorer_explanation(summary: Mapping[str, Any], heatmap_mode: str, records_in_heatmap: int) -> tuple[str, str, str]:
    mode = "quantidades" if heatmap_mode == "Contagem absoluta" else "percentuais por posição de largada"
    finding = (
        f"Os filtros mostram {_fmt_count(summary.get('registros'))} resultados em "
        f"{_fmt_count(summary.get('eventos'))} corridas. O mapa apresenta {mode} e reúne {_fmt_count(records_in_heatmap)} resultados nas células."
    )
    return (
        "Escolha temporadas, corridas, pilotos, equipes e posições para explorar um recorte. Os indicadores e o mapa acompanham os filtros.",
        finding,
        "Cada percentual compara resultados da mesma posição de largada. Em grupos pequenos, poucas corridas podem mudar bastante a proporção."
    )


def historical_explanation(lap_stats: Mapping[str, Any], sensitivity: pd.DataFrame) -> tuple[str, str, str]:
    mean = _finite(lap_stats.get("media_diferenca_seg"))
    median = _finite(lap_stats.get("mediana_diferenca_seg"))
    pairs = lap_stats.get("pares")
    direction = _result_direction(
        mean,
        positive="Em média, a volta registrada foi mais lenta em 2014.",
        negative="Em média, a volta registrada foi mais rápida em 2014.",
        neutral="A média das diferenças foi zero.",
    )
    result = (
        f"Nos {_fmt_count(pairs)} pares, a média foi {_fmt_signed(mean, 3, ' s')} e o valor central (mediana), "
        f"{_fmt_signed(median, 3, ' s')}. {direction}"
    )
    japan = sensitivity.loc[sensitivity["cenario"].eq("Sem Japão")] if "cenario" in sensitivity else pd.DataFrame()
    full = sensitivity.loc[sensitivity["cenario"].eq("Amostra completa")] if "cenario" in sensitivity else pd.DataFrame()
    if not japan.empty and not full.empty:
        without_mean = _finite(japan.iloc[0].get("media_seg"))
        full_mean = _finite(full.iloc[0].get("media_seg"))
        if without_mean is not None and full_mean is not None:
            change = without_mean - full_mean
            remaining = _result_direction(
                without_mean,
                positive="continua positiva",
                negative="passa a negativa",
                neutral="chega a zero",
            )
            result += f" Sem Japão, a média mudou {_fmt_signed(change, 3, ' s')} e {remaining.lower()}"
    return (
        "Comparamos a volta mais rápida registrada do mesmo piloto, equipe e circuito em 2013 e 2014.",
        result,
        "A comparação não separa efeitos de pista, clima, estratégia ou regulamento. Há uma inconsistência documentada na Austrália 2014; a amostra não foi validada externamente."
    )


def regression_explanation(reg: Mapping[str, Any], refs: Mapping[str, Any]) -> tuple[str, str, str]:
    mae = _finite(reg.get("mae"))
    baseline = _finite(refs.get("mae_previsao_posicao_igual_ao_grid"))
    if mae is None:
        finding = "O erro médio do modelo não está disponível neste teste."
    elif baseline is None:
        finding = f"O modelo errou, em média, {_fmt(mae, 1)} posições. A comparação com a regra simples não está disponível."
    elif mae > baseline:
        finding = (
            f"O modelo errou, em média, {_fmt(mae, 1)} posições. "
            f"A regra simples baseada no grid teve erro um pouco menor: {_fmt(baseline, 1)} posições."
        )
    elif mae < baseline:
        finding = (
            f"O modelo errou, em média, {_fmt(mae, 1)} posições. "
            f"Neste teste, foi melhor que a regra simples baseada no grid ({_fmt(baseline, 1)} posições de erro)."
        )
    else:
        finding = f"Modelo e regra baseada no grid erraram, em média, {_fmt(mae, 1)} posições neste teste."
    return (
        "O modelo tentou estimar a posição final registrada usando a posição de largada.",
        finding,
        "É um teste com temporadas passadas. Não garante o resultado de uma corrida futura nem de um piloto específico."
    )


def classification_explanation(clf: Mapping[str, Any]) -> tuple[str, str, str]:
    if clf.get("erro_classificador"):
        return (
            "O modelo tenta identificar resultados entre os três primeiros.",
            "O classificador não pôde ser avaliado com os dados disponíveis.",
            "Métrica indisponível não significa zero nem desempenho ruim quantificado."
        )
    precision, recall = (_finite(clf.get(key)) for key in ("precision", "recall"))
    parts = []
    if recall is not None:
        parts.append(f"De cada 100 pódios reais, o modelo identifica cerca de {_fmt_percent(recall, 0)}.")
    else:
        parts.append("A proporção de pódios reais identificados não está disponível.")
    if precision is not None:
        parts.append(f"De cada 100 previsões de pódio, cerca de {_fmt_percent(precision, 0)} se confirmam.")
    else:
        parts.append("A proporção de previsões de pódio que se confirmam não está disponível.")
    return (
        "O modelo tenta identificar se um resultado ficará entre os três primeiros.",
        " ".join(parts),
        "São proporções observadas no teste, reescaladas para facilitar a leitura; não são novos testes com exatamente 100 casos. O teste histórico não garante desempenho futuro."
    )


def climate_explanation(observations: pd.DataFrame, events: pd.DataFrame) -> tuple[str, str, str]:
    tmp_count = int(pd.to_numeric(observations.get("air_temperature_c"), errors="coerce").notna().sum())
    dew_count = int(pd.to_numeric(observations.get("dew_point_c"), errors="coerce").notna().sum())
    return (
        "A estação mede o ar onde está, não na pista. O ponto de orvalho ajuda a descrever a umidade.",
        f"Há {_fmt_count(len(observations))} observações em {_fmt_count(len(events))} janelas, com {_fmt_count(tmp_count)} temperaturas e {_fmt_count(dew_count)} pontos de orvalho numéricos.",
        "A aprovação de qualidade não garante que a estação represente o circuito. Os horários são programados, não a largada real. Sem horários de clima pareados às voltas, não podemos explicar os tempos dos pilotos."
    )


def methodology_explanation(report: Mapping[str, Any] | None) -> tuple[str, str, str]:
    if report is None:
        finding = "A auditoria detalhada não está disponível agora."
    else:
        validation = report.get("validation", {})
        finding = (
            f"A auditoria verificou {_fmt_count(len(report.get('tables', {})))} tabelas: "
            f"{_fmt_count(validation.get('blocking_issues', 0))} problemas bloqueantes e "
            f"{_fmt_count(validation.get('warning_issues', 0))} avisos."
        )
    return (
        "Aqui explicamos quais dados entram na análise e como são verificados.",
        finding + " Registros ausentes e inconsistências são mantidos, não corrigidos em silêncio.",
        "A origem e a licença dos dados ainda não estão plenamente documentadas. A base histórica também não foi validada externamente por completo."
    )


def render_explanation(mode: str, explanation: tuple[str, str, str]) -> None:
    question, finding, limitation = explanation
    if mode == SIMPLE_MODE:
        st.markdown(
            f"**O que estamos analisando?** {question}\n\n"
            f"**O que descobrimos?** {finding}\n\n"
            f"**O que isso não permite concluir?** {limitation}"
        )
    else:
        st.markdown(
            f"**Leitura técnica — pergunta e unidade de análise:** {question}\n\n"
            f"**Resultado condicionado aos dados disponíveis:** {finding}\n\n"
            f"**Limites de interpretação:** {limitation}"
        )


def render_glossary() -> None:
    """One compact glossary shared by all areas."""
    with st.expander("Glossário educativo", expanded=False):
        st.markdown("\n".join(f"- **{term}:** {definition}" for term, definition in GLOSSARY))
