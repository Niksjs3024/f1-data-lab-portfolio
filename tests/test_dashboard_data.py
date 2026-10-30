"""Testes unitarios de apresentacao de dados (src/dashboard/data.py).

Nao dependem dos CSVs históricos: constroem o DataFrame a partir das tabelas
sinteticas do conftest, nao do pipeline batch.
"""

import pandas as pd
import pytest

from src.dashboard import data as dashboard_data
from src.dashboard.data import (
    apply_explorer_filters,
    constructor_options,
    distinct_grid_options,
    driver_options,
    grid_finish_crosstab,
    heatmap_percentages,
    race_options,
    scope_summary,
)
from src.pipeline import build_dataset


@pytest.fixture
def df(synthetic_tables):
    return build_dataset(synthetic_tables)


def test_distinct_grid_options_sem_corte_em_p20(df):
    """Posicoes observadas, incluindo acima de P20, sao retornadas inteiras."""
    opts = distinct_grid_options(df)
    assert opts == sorted(int(g) for g in pd.unique(df["grid"].dropna()) if int(g) > 0)
    assert all(g > 0 for g in opts)


def test_apply_explorer_filters_nao_mutacao_entrada(df):
    """Filtros devolvem um recorte sem alterar o DataFrame de entrada."""
    antes = df.copy()
    out = apply_explorer_filters(df, years=[2014], races=[10], drivers=[], constructors=[], grids=[1, 2])
    assert out is not df
    pd.testing.assert_frame_equal(df, antes)


def test_apply_explorer_filters_combinados(df):
    out = apply_explorer_filters(df, years=[2014], races=[10], drivers=[], constructors=[], grids=[1, 2])
    esperados = df[(df["year"] == 2014) & (df["race_id"] == 10) & (df["grid"].isin([1, 2]))]
    pd.testing.assert_frame_equal(out.reset_index(drop=True), esperados.reset_index(drop=True))


def test_apply_explorer_filters_vazio_significa_todos(df):
    out = apply_explorer_filters(df, years=[], races=[], drivers=[], constructors=[], grids=[])
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))


def test_apply_explorer_filters_sem_intersecao_estado_vazio(df):
    out = apply_explorer_filters(df, years=[1999], races=[], drivers=[], constructors=[], grids=[99])
    assert out.empty


def test_scope_summary_registros_versus_eventos(df):
    s = scope_summary(df)
    assert s["registros"] == len(df)
    assert s["eventos"] == int(df["race_id"].nunique())
    assert s["pilotos"] == int(df["driver_id"].nunique())
    assert s["construtoras"] == int(df["constructor_id"].nunique())
    assert s["temporadas"] == int(df["year"].nunique())


def test_grid_finish_crosstab_soma_todos_os_registros_do_recorte(df):
    """A matriz completa preserva cada registro: nada e descartado por ausencia de celula."""
    counts = grid_finish_crosstab(df)
    assert int(counts.values.sum()) == len(df)
    assert list(counts.index) == distinct_grid_options(df)
    assert list(counts.columns) == sorted(
        int(p) for p in pd.unique(df["finish_position"].dropna()) if int(p) > 0
    )


def test_grid_finish_crosstab_recorte_filtrado(df):
    recorte = apply_explorer_filters(df, years=[2014], races=[10], drivers=[], constructors=[], grids=[])
    counts = grid_finish_crosstab(recorte)
    assert int(counts.values.sum()) == len(recorte) == 3


def test_grid_finish_crosstab_inclui_posicoes_acima_de_p20():
    """Dominio vem dos dados: nao existe corte artificial em P20."""
    recorte = pd.DataFrame({"grid": [1, 25], "finish_position": [1, 20]})
    counts = grid_finish_crosstab(recorte)
    assert list(counts.index) == [1, 25]
    assert list(counts.columns) == [1, 20]
    assert int(counts.values.sum()) == 2


def test_grid_finish_crosstab_vazio(df):
    assert grid_finish_crosstab(df.iloc[0:0]).empty


def test_heatmap_percentages_linhas_com_registros_somam_100(df):
    counts = grid_finish_crosstab(df)
    pct = heatmap_percentages(counts)
    for grid, total in counts.sum(axis=1).items():
        if total > 0:
            assert abs(float(pct.loc[grid].sum()) - 100.0) < 1e-9


def test_heatmap_percentages_usa_denominador_do_grid_e_nao_o_total_geral():
    """Cada linha e normalizada pelo proprio total (4), nao pela soma geral (8)."""
    counts = pd.DataFrame([[2, 2], [1, 3]], index=[1, 2], columns=[1, 2])
    pct = heatmap_percentages(counts)
    assert pct.loc[1].tolist() == [50.0, 50.0]
    assert pct.loc[2].tolist() == [25.0, 75.0]
    assert abs(float(pct.values.sum()) - 200.0) < 1e-9


def test_heatmap_percentages_linha_sem_registros_e_indefinida():
    """Linha vazia nao vira zero: percentual indefinido permanece NaN."""
    counts = pd.DataFrame([[0, 0], [1, 3]], index=[1, 2], columns=[1, 2])
    pct = heatmap_percentages(counts)
    assert pct.loc[1].isna().all()
    assert pct.loc[2].tolist() == [25.0, 75.0]


def test_heatmap_percentages_matriz_vazia():
    assert heatmap_percentages(pd.DataFrame()).empty


def test_opcoes_chaveadas_por_id_sem_duplicar(df):
    for opcao in (race_options(df), driver_options(df), constructor_options(df)):
        ids = [item[0] for item in opcao]
        assert len(ids) == len(set(ids)), f"IDs duplicados em {opcao}"
        assert all(isinstance(i, int) for i in ids)
    rotulos_races = {rotulo for _, rotulo in race_options(df)}
    assert len(rotulos_races) == len(race_options(df))


def test_load_climate_bundle_reads_versioned_local_files():
    bundle = dashboard_data.load_climate_bundle(dashboard_data.climate_file_signature())
    assert len(bundle["observations"]) == 90
    assert len(bundle["events"]) == 10
    assert set(bundle["observations"]["race_id"].astype(str)) <= set(bundle["events"]["race_id"].astype(str))
    assert int(bundle["observations"]["tmp_qc_eligible"].map(str).str.casefold().eq("true").sum()) == 90


def test_climate_signature_changes_when_a_local_file_changes(tmp_path, monkeypatch):
    paths = [tmp_path / f"local-{index}.csv" for index in range(5)]
    for path in paths:
        path.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(dashboard_data, "ROOT", tmp_path)
    for name, path in zip((
        "CLIMATE_EVENTS_PATH", "CLIMATE_EVIDENCE_PATH", "CLIMATE_OBSERVATIONS_PATH",
        "CLIMATE_EVENT_SUMMARY_PATH", "CLIMATE_REPORT_TYPE_SUMMARY_PATH",
    ), paths):
        monkeypatch.setattr(dashboard_data, name, path)
    before = dashboard_data.climate_file_signature()
    paths[2].write_text("changed size", encoding="utf-8")
    assert dashboard_data.climate_file_signature() != before
