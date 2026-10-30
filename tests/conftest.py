"""Fixtures sintéticos: nenhum teste depende dos CSVs históricos."""
import importlib

import pandas as pd
import pytest


@pytest.fixture
def quality_api():
    # Importação tardia mantém a coleta dos testes possível enquanto a API é criada.
    return importlib.import_module("src.data_quality")


@pytest.fixture
def synthetic_tables():
    """Duas corridas, três participantes por corrida e todas as dimensões válidas."""
    return {
        "results": pd.DataFrame({
            "resultId": [101, 102, 103, 104, 105, 106],
            "raceId": [10, 10, 10, 11, 11, 11],
            "driverId": [1, 2, 3, 1, 2, 3],
            "constructorId": [7, 7, 8, 7, 7, 8],
            "statusId": [1, 1, 2, 1, 1, 2],
            "grid": [1, 2, 3, 1, 2, 3],
            "positionOrder": [1, 2, 19, 2, 1, 18],
            "position": ["1", "2", None, "2", "1", None],
            "positionText": ["1", "2", "R", "2", "1", "R"],
            "points": [25, 18, 0, 18, 25, 0],
            "laps": [58, 58, 12, 60, 60, 9],
            "fastestLapTime": ["1:25.0", "1:26.0", "1:29.0",
                               "1:24.0", "1:25.0", "1:28.0"],
            "fastestLapSpeed": [215, 214, 210, 216, 215, 211],
        }),
        "races": pd.DataFrame({
            "raceId": [10, 11], "year": [2014, 2015],
            "circuitId": [20, 21], "name": ["GP A", "GP B"],
        }),
        "drivers": pd.DataFrame({
            "driverId": [1, 2, 3], "forename": ["Ana", "Beto", "Caio"],
            "surname": ["Um", "Dois", "Três"],
        }),
        "constructors": pd.DataFrame({
            "constructorId": [7, 8], "name": ["Equipe A", "Equipe B"],
        }),
        "status": pd.DataFrame({
            "statusId": [1, 2, 3, 4],
            "status": ["Finished", "Engine", "Disqualified", "Withdrawn"],
        }),
    }


@pytest.fixture
def set_cell():
    """Permite testar strings/NA em colunas numéricas sem depender de coerção pandas."""
    def set_value(tables, table, column, value, row=0):
        tables[table][column] = tables[table][column].astype(object)
        tables[table].at[row, column] = value

    return set_value
