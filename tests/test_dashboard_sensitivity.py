import pandas as pd
import pytest

from src.dashboard.sensitivity import historical_sensitivity_summary


def test_historical_sensitivity_keeps_full_sample_and_recomputes_each_gp():
    pairs = pd.DataFrame({
        "grand_prix": ["Japanese Grand Prix", "Japanese Grand Prix", "Brazilian Grand Prix", "Brazilian Grand Prix"],
        "diferenca_2014_menos_2013_seg": [1.0, 3.0, -1.0, 5.0],
        "driver": ["A", "B", "C", "D"],
    })
    original = pairs.copy(deep=True)

    summary = historical_sensitivity_summary(pairs).set_index("cenario")

    assert summary.loc["Amostra completa", "pares"] == 4
    assert summary.loc["Amostra completa", "media_seg"] == 2.0
    assert summary.loc["Amostra completa", "mediana_seg"] == 2.0
    assert summary.loc["Sem Japão", "pares"] == 2
    assert summary.loc["Sem Japão", "media_seg"] == 2.0
    assert summary.loc["Sem Japão", "mediana_seg"] == 2.0
    assert summary.loc["Sem Brazilian Grand Prix", "pares"] == 2
    assert summary.loc["Sem Brazilian Grand Prix", "media_seg"] == 2.0
    pd.testing.assert_frame_equal(pairs, original)


def test_historical_sensitivity_handles_empty_and_non_numeric_pairs():
    pairs = pd.DataFrame({
        "grand_prix": ["Japanese Grand Prix"],
        "diferenca_2014_menos_2013_seg": ["not-a-number"],
    })

    summary = historical_sensitivity_summary(pairs)

    assert summary.iloc[0]["pares"] == 0
    assert pd.isna(summary.iloc[0]["media_seg"])
    assert pd.isna(summary.iloc[0]["mediana_seg"])
    assert summary.iloc[1]["cenario"] == "Sem Japão"
    assert summary.iloc[1]["pares"] == 0


def test_historical_sensitivity_requires_pair_columns():
    with pytest.raises(ValueError, match="Colunas ausentes"):
        historical_sensitivity_summary(pd.DataFrame({"grand_prix": ["GP"]}))
