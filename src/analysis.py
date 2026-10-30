"""Estatísticas e comparação descritiva. Nenhum indicador usa números fixos."""
import pandas as pd
from .pipeline import parse_lap_seconds
from .data_quality import numeric_values, validate_analysis_frame


def grid_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Percentuais por registro de grid; mantém o esquema histórico de seis colunas.

    Pontos ausentes não são zero: pontos_pct fica NaN quando qualquer registro
    do grupo tiver pontuação desconhecida. Nenhum denominador é trocado.
    """
    validate_analysis_frame(df)
    values = df.copy()
    for column in ("grid", "finish_position", "points"):
        values[column] = numeric_values(values[column])
    summary = (values.assign(
        venceu=values["finish_position"].eq(1),
        podio=values["finish_position"].between(1, 3),
        top10=values["finish_position"].between(1, 10),
        pontuou=values["points"].gt(0),
    ).groupby("grid", as_index=False).agg(
        corridas=("result_id", "size"),
        vitoria_pct=("venceu", "mean"),
        podio_pct=("podio", "mean"),
        top10_pct=("top10", "mean"),
        pontos_pct=("pontuou", "mean"),
    ))
    unknown = values["points"].isna().groupby(values["grid"]).any()
    summary.loc[summary["grid"].map(unknown).fillna(False), "pontos_pct"] = float("nan")
    for col in ("vitoria_pct", "podio_pct", "top10_pct", "pontos_pct"):
        summary[col] = (100 * summary[col]).round(2)
    return summary.sort_values("grid").reset_index(drop=True)


def compare_2013_2014(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compara a volta mais rápida do mesmo piloto/equipe/circuito nos dois anos.

    Exclui pares sem volta registrada. Não permite inferência causal.
    """
    subset = df.loc[df["year"].isin([2013, 2014]), [
        "year", "driver_id", "constructor_id", "circuit_id", "grand_prix",
        "driver", "constructor", "fastest_lap_time"
    ]].copy()
    subset["lap_seconds"] = subset["fastest_lap_time"].map(parse_lap_seconds)
    subset = subset.dropna(subset=["lap_seconds"])
    keys = ["driver_id", "constructor_id", "circuit_id"]
    wide = subset.pivot_table(index=keys, columns="year", values="lap_seconds", aggfunc="mean")
    if not {2013, 2014}.issubset(wide.columns):
        return pd.DataFrame(), {"pares": 0, "media_diferenca_seg": None, "mediana_diferenca_seg": None}
    paired = wide.dropna(subset=[2013, 2014]).reset_index()
    paired = paired.rename(columns={2013: "volta_2013_seg", 2014: "volta_2014_seg"})
    paired["diferenca_2014_menos_2013_seg"] = paired["volta_2014_seg"] - paired["volta_2013_seg"]
    labels = (subset.sort_values("year").drop_duplicates(keys)[keys + ["grand_prix", "driver", "constructor"]])
    paired = paired.merge(labels, on=keys, how="left", validate="one_to_one")
    metrics = {
        "pares": int(len(paired)),
        "media_diferenca_seg": (round(float(paired["diferenca_2014_menos_2013_seg"].mean()), 3)
                                 if len(paired) else None),
        "mediana_diferenca_seg": (round(float(paired["diferenca_2014_menos_2013_seg"].median()), 3)
                                   if len(paired) else None),
    }
    return paired, metrics
