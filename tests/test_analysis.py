import math
import pandas as pd
from src.pipeline import parse_lap_seconds, build_dataset
from src.analysis import grid_probabilities


def test_parse_lap_seconds():
    assert parse_lap_seconds("01:27.5") == 87.5
    assert parse_lap_seconds("1:02:03.2") == 3723.2
    assert math.isnan(parse_lap_seconds(None))
    assert math.isnan(parse_lap_seconds("N/A"))


def test_probabilities_not_fixed_230():
    df = pd.DataFrame({
        "grid": [1, 1, 2, 2, 2],
        "result_id": [1, 2, 3, 4, 5],
        "finish_position": [1, 11, 1, 2, 10],
        "points": [25, 0, 25, 18, 1],
    })
    result = grid_probabilities(df).set_index("grid")
    assert result.loc[1, "corridas"] == 2
    assert result.loc[1, "vitoria_pct"] == 50
    assert result.loc[2, "podio_pct"] == 66.67


def test_preserves_missing_fastest_lap_and_retirements():
    tables = {
        "results": pd.DataFrame({"resultId": [1, 2], "raceId": [10, 10], "driverId": [1, 2],
                                 "constructorId": [7, 7], "statusId": [1, 2],
                                 "grid": [1, 2], "positionOrder": [1, 19],
                                 "points": [25, 0], "fastestLapTime": ["1:25.0", None],
                                 "fastestLapSpeed": [215, None]}),
        "races": pd.DataFrame({"raceId": [10], "year": [2014], "circuitId": [20], "name": ["GP"]}),
        "drivers": pd.DataFrame({"driverId": [1, 2], "forename": ["A", "B"], "surname": ["One", "Two"]}),
        "constructors": pd.DataFrame({"constructorId": [7], "name": ["Team"]}),
        "status": pd.DataFrame({"statusId": [1, 2], "status": ["Finished", "Engine"]}),
    }
    df = build_dataset(tables)
    assert len(df) == 2
    assert df["fastest_lap_time"].isna().sum() == 1
    assert "Engine" in df["status"].tolist()
