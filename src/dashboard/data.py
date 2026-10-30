"""Acesso a dados e filtros do dashboard (apenas adaptações de apresentação).

Não replica regras estatísticas: frequências, pareamento histórico e modelos
continuam vivendo em src/analysis.py, src/model.py e src/pipeline.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import pipeline
from src.pipeline import build_dataset

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CLIMATE_EVENTS_PATH = ROOT / "data" / "climate_phase3b_20261022" / "events.csv"
CLIMATE_EVIDENCE_PATH = ROOT / "data" / "climate_phase3b_20261022" / "race_evidence.csv"
CLIMATE_OBSERVATIONS_PATH = ROOT / "data" / "climate_phase3c_20261022" / "noaa_observations_quality.csv"
CLIMATE_EVENT_SUMMARY_PATH = ROOT / "outputs" / "climate_phase3c_20261022" / "quality_summary_by_event.csv"
CLIMATE_REPORT_TYPE_SUMMARY_PATH = ROOT / "outputs" / "climate_phase3c_20261022" / "quality_summary_common_report_types.csv"
CLIMATE_PRESENTATION_REVISION = "f1-climate-dashboard-1"

# Versão do contrato de apresentação. Mudar quando a lógica de filtros ou
# adaptação de dados da UI for alterada, para invalidar caches relacionados.
PRESENTATION_REVISION = "f1-dashboard-1"


def file_signature(raw_dir: Path = RAW_DIR) -> str:
    """Assinatura barata dos cinco CSVs para invalidação de cache.

    Caminho, existência, tamanho e mtime_ns. Não é prova criptográfica de
    integridade; serve apenas para evitar recarregamentos desnecessários.
    """
    digest = hashlib.sha256()
    digest.update(PRESENTATION_REVISION.encode())
    for name in ("results", "races", "drivers", "constructors", "status"):
        path = raw_dir / f"{name}.csv"
        digest.update(name.encode())
        if path.exists():
            stat = path.stat()
            digest.update(f":{stat.st_size}:{int(stat.st_mtime_ns)}".encode())
        else:
            digest.update(":MISSING".encode())
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def load_app_data(raw_dir: str, signature: str) -> pd.DataFrame:
    """Carrega e valida a base integrada (padrão 2004–2017, grid > 0)."""
    return build_dataset(pipeline.load_data(Path(raw_dir)))


def climate_file_signature() -> str:
    """Assinatura isolada dos arquivos climáticos locais para invalidar o cache."""
    digest = hashlib.sha256(CLIMATE_PRESENTATION_REVISION.encode())
    for path in (
        CLIMATE_EVENTS_PATH,
        CLIMATE_EVIDENCE_PATH,
        CLIMATE_OBSERVATIONS_PATH,
        CLIMATE_EVENT_SUMMARY_PATH,
        CLIMATE_REPORT_TYPE_SUMMARY_PATH,
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        if path.exists():
            stat = path.stat()
            digest.update(f":{stat.st_size}:{int(stat.st_mtime_ns)}".encode())
        else:
            digest.update(b":MISSING")
    return digest.hexdigest()


def _read_climate_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Esquema incompatível em {path.name}: faltam {', '.join(sorted(missing))}")
    return frame


@st.cache_data(show_spinner=False)
def load_climate_bundle(signature: str) -> dict[str, pd.DataFrame]:
    """Carrega apenas os arquivos climáticos versionados, sem acesso à rede."""
    observations = _read_climate_csv(CLIMATE_OBSERVATIONS_PATH, {
        "season", "grand_prix", "race_id", "scheduled_start_utc", "observation_time_utc",
        "report_type", "distance_to_circuit_km", "air_temperature_c", "dew_point_c",
        "station_name", "tmp_quality_flag", "dew_quality_flag", "tmp_quality_class", "dew_quality_class",
        "tmp_quality_description", "dew_quality_description", "tmp_qc_eligible", "dew_qc_eligible",
    })
    events = _read_climate_csv(CLIMATE_EVENTS_PATH, {
        "season", "grand_prix", "race_id", "scheduled_start_utc", "scheduled_start_local",
        "timezone", "utc_offset",
    })
    evidence = _read_climate_csv(CLIMATE_EVIDENCE_PATH, {
        "season", "grand_prix", "race_id", "session", "fact_type", "description",
        "source_class", "source_url", "confidence", "limitation",
    })
    event_summary = _read_climate_csv(CLIMATE_EVENT_SUMMARY_PATH, {
        "season", "grand_prix", "race_id", "scheduled_start_utc", "observation_count",
        "unique_timestamp_count", "first_observation_utc", "last_observation_utc", "report_type_counts",
        "distance_to_circuit_km", "tmp_numeric_count_all", "tmp_mean_all_c", "tmp_qc_accepted_count",
        "tmp_mean_qc_accepted_c", "dew_numeric_count_all", "dew_mean_all_c",
        "dew_qc_accepted_count", "dew_mean_qc_accepted_c",
    })
    common_types = _read_climate_csv(CLIMATE_REPORT_TYPE_SUMMARY_PATH, {
        "grand_prix", "report_type", "comparison_scope",
    })
    if observations["race_id"].isna().any() or not set(observations["race_id"].astype(str)) <= set(events["race_id"].astype(str)):
        raise ValueError("Observações climáticas referenciam eventos ausentes ou inválidos")
    return {
        "observations": observations,
        "events": events,
        "evidence": evidence,
        "event_summary": event_summary,
        "common_types": common_types,
    }


@st.cache_data(show_spinner=False)
def load_audit_report(raw_dir: str, signature: str) -> dict[str, Any]:
    """Auditoria completa dos arquivos atuais, calculada em memória."""
    from src.audit_data import create_audit_report

    return create_audit_report(Path(raw_dir))


@st.cache_data(show_spinner=False)
def evaluate_model_experiment(df: pd.DataFrame, signature: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Avaliação temporal de ML (treino ≤ 2014, teste ≥ 2015), cacheada.

    Recebe o DataFrame padrão para que filtros exploratórios nunca acionem
    novo treinamento.
    """
    from src.model import evaluate_models_with_arrays

    return evaluate_models_with_arrays(df)


def distinct_grid_options(df: pd.DataFrame) -> list[int]:
    """Posições de grid observadas no recorte, incluindo acima de P20."""
    return sorted(int(g) for g in pd.unique(df["grid"].dropna()) if int(g) > 0)


def race_options(df: pd.DataFrame) -> list[tuple[int, str]]:
    """Eventos rastreáveis por race_id, nunca apenas pelo nome do GP."""
    frames = (
        df[["race_id", "year", "grand_prix"]]
        .drop_duplicates("race_id")
        .sort_values(["year", "race_id"])
    )
    return [
        (int(row.race_id), f"{int(row.year)} — {row.grand_prix} (id={int(row.race_id)})")
        for row in frames.itertuples(index=False)
    ]


def driver_options(df: pd.DataFrame) -> list[tuple[int, str]]:
    frames = (
        df[["driver_id", "driver"]]
        .drop_duplicates("driver_id")
        .sort_values("driver")
    )
    return [(int(row.driver_id), row.driver) for row in frames.itertuples(index=False)]


def constructor_options(df: pd.DataFrame) -> list[tuple[int, str]]:
    frames = (
        df[["constructor_id", "constructor"]]
        .drop_duplicates("constructor_id")
        .sort_values("constructor")
    )
    return [(int(row.constructor_id), row.constructor) for row in frames.itertuples(index=False)]


def apply_explorer_filters(
    df: pd.DataFrame,
    years: list[int] | None,
    races: list[int] | None,
    drivers: list[int] | None,
    constructors: list[int] | None,
    grids: list[int] | None,
) -> pd.DataFrame:
    """Aplica filtros exploratórios sobre uma cópia, sem alterar a entrada."""
    filtered = df.copy()
    if years:
        filtered = filtered[filtered["year"].isin(years)]
    if races:
        filtered = filtered[filtered["race_id"].isin(races)]
    if drivers:
        filtered = filtered[filtered["driver_id"].isin(drivers)]
    if constructors:
        filtered = filtered[filtered["constructor_id"].isin(constructors)]
    if grids:
        filtered = filtered[filtered["grid"].isin(grids)]
    return filtered


def grid_finish_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz grid (linhas) × ordenação final registrada (colunas).

    O domínio é derivado dos dados observados, sem corte em P20. Combinações
    ausentes entram como zero, de modo que a soma das células é o total de
    registros com grid e ordenação registrados no recorte.
    """
    if df.empty:
        return pd.DataFrame()
    full_grid = sorted(int(g) for g in pd.unique(df["grid"].dropna()) if int(g) > 0)
    full_pos = sorted(int(p) for p in pd.unique(df["finish_position"].dropna()) if int(p) > 0)
    if not full_grid or not full_pos:
        return pd.DataFrame()
    counts = pd.crosstab(df["grid"], df["finish_position"])
    return counts.reindex(index=full_grid, columns=full_pos, fill_value=0)


def heatmap_percentages(counts: pd.DataFrame) -> pd.DataFrame:
    """Percentual de cada célula sobre o total de registros do seu próprio grid.

    O denominador é o total da linha (registros analisados naquela posição de
    largada), nunca o total geral do recorte. Linha sem registros resulta em
    `NaN`: percentual indefinido não é zero.
    """
    if counts.empty:
        return counts.astype(float)
    totals = counts.sum(axis=1).astype(float)
    denominador = totals.where(totals > 0)
    return counts.astype(float).div(denominador, axis=0) * 100.0


def scope_summary(df: pd.DataFrame) -> dict[str, int | str]:
    """Contagens do recorte, distinguindo registros de eventos distintos."""
    return {
        "registros": int(len(df)),
        "eventos": int(df["race_id"].nunique()) if "race_id" in df else 0,
        "pilotos": int(df["driver_id"].nunique()) if "driver_id" in df else 0,
        "construtoras": int(df["constructor_id"].nunique()) if "constructor_id" in df else 0,
        "temporadas": int(df["year"].nunique()) if "year" in df else 0,
    }
