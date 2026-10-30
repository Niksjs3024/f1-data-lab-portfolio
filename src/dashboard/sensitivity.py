"""Sensitivity summaries for the existing historical comparison in the dashboard."""

from __future__ import annotations

import pandas as pd

DIFFERENCE_COLUMN = "diferenca_2014_menos_2013_seg"


def historical_sensitivity_summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize the full pair set and each leave-one-Grand-Prix-out scenario."""
    required = {"grand_prix", DIFFERENCE_COLUMN}
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"Colunas ausentes para sensibilidade: {', '.join(sorted(missing))}")

    values = paired[["grand_prix", DIFFERENCE_COLUMN]].copy()
    values[DIFFERENCE_COLUMN] = pd.to_numeric(values[DIFFERENCE_COLUMN], errors="coerce")
    gps = sorted(values.loc[values["grand_prix"].notna(), "grand_prix"].astype(str).unique())
    values = values.dropna(subset=["grand_prix", DIFFERENCE_COLUMN])

    def metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
        observed = frame[DIFFERENCE_COLUMN]
        return {
            "pares": int(observed.count()),
            "media_seg": round(float(observed.mean()), 3) if observed.count() else None,
            "mediana_seg": round(float(observed.median()), 3) if observed.count() else None,
        }

    full = metrics(values)
    rows = [{"cenario": "Amostra completa", "gp_excluido": "Nenhum", **full}]
    for gp in gps:
        remaining = values.loc[values["grand_prix"].astype(str).ne(gp)]
        rows.append({"cenario": f"Sem {gp}", "gp_excluido": gp, **metrics(remaining)})

    # Nomeia o cenário explicitamente solicitado sem depender de um apelido traduzido.
    japan_rows = [
        row for row in rows[1:]
        if any(token in str(row["gp_excluido"]).casefold() for token in ("japan", "japão"))
    ]
    for row in japan_rows:
        row["cenario"] = "Sem Japão"
    for row in rows:
        row["delta_media_vs_completa_seg"] = (
            round(row["media_seg"] - full["media_seg"], 3)
            if row["media_seg"] is not None and full["media_seg"] is not None else None
        )
        row["delta_mediana_vs_completa_seg"] = (
            round(row["mediana_seg"] - full["mediana_seg"], 3)
            if row["mediana_seg"] is not None and full["mediana_seg"] is not None else None
        )
    return pd.DataFrame(rows)
