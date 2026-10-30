"""Pipeline reprodutível de preparação dos resultados históricos da Fórmula 1."""
from pathlib import Path
import pandas as pd
from .data_quality import REQUIRED_FILES, numeric_values, parse_lap_seconds, validate_tables


def read_csv(path: Path) -> pd.DataFrame:
    """Tenta UTF-8 e Latin-1 sem reparar grafias ou reescrever o original."""
    try:
        frame = pd.read_csv(path, encoding="utf-8", low_memory=False)
        encoding = "utf-8"
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)
        encoding = "ISO-8859-1"
    frame.attrs["source_encoding"] = encoding
    return frame


def load_data(raw_dir: Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    missing = [name for name in REQUIRED_FILES if not (raw_dir / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes em {raw_dir}: {', '.join(missing)}")
    return {name: read_csv(raw_dir / f"{name}.csv") for name in REQUIRED_FILES}


def build_dataset(tables: dict[str, pd.DataFrame], start_year: int = 2004,
                  end_year: int = 2017, *, strict: bool = False) -> pd.DataFrame:
    """Preserva abandonos, DSQs e resultados sem volta rápida.

    O modo padrão bloqueia falhas de contrato, mas registra anomalias históricas
    sem descartá-las. strict=True também reprova avisos de auditoria. Grid é o
    campo de largada informado pela fonte; grid positivo não prova participação.
    finish_position continua sendo positionOrder, não posição oficial garantida.
    """
    report = validate_tables(tables, start_year, end_year, strict=strict)
    # Normaliza chaves em cópias: evita incompatibilidade int/string nas junções.
    prepared = {name: table.copy() for name, table in tables.items()}
    keys = {"results": ["resultId", "raceId", "driverId", "constructorId", "statusId"],
            "races": ["raceId", "circuitId"], "drivers": ["driverId"],
            "constructors": ["constructorId"], "status": ["statusId"]}
    for name, columns in keys.items():
        for column in columns:
            prepared[name][column] = numeric_values(prepared[name][column])
    results = prepared["results"].copy()
    # Campos opcionais permitem manter compatibilidade com fixtures e bases mínimas.
    for column in ("laps", "position", "positionText", "fastestLapTime", "fastestLapSpeed"):
        if column not in results:
            results[column] = float("nan")
    results = results[[
        "resultId", "raceId", "driverId", "constructorId", "statusId",
        "grid", "positionOrder", "points", "laps", "position", "positionText",
        "fastestLapTime", "fastestLapSpeed"
    ]]
    dimensions = [
        ("races", "raceId", prepared["races"][["raceId", "year", "circuitId", "name"]].rename(columns={"name": "grand_prix"})),
        ("drivers", "driverId", prepared["drivers"][["driverId", "forename", "surname"]]),
        ("constructors", "constructorId", prepared["constructors"][["constructorId", "name"]].rename(columns={"name": "constructor"})),
        ("status", "statusId", prepared["status"][["statusId", "status"]]),
    ]
    df = results
    joins = []
    for name, key, dimension in dimensions:
        before = len(df)
        df = df.merge(dimension, on=key, how="left", validate="many_to_one", indicator="_join")
        if len(df) != before or not df["_join"].eq("both").all():
            raise ValueError(f"Integridade da junção com {name} violada; nenhum descarte silencioso permitido.")
        joins.append({"table": name, "before": before, "after": len(df), "unmatched": 0})
        df = df.drop(columns="_join")

    df = df.rename(columns={
        "resultId": "result_id", "raceId": "race_id", "driverId": "driver_id",
        "constructorId": "constructor_id", "statusId": "status_id",
        "circuitId": "circuit_id", "positionOrder": "finish_position",
        "positionText": "position_text", "fastestLapTime": "fastest_lap_time",
        "fastestLapSpeed": "fastest_lap_speed"
    })
    df["driver"] = df["forename"].fillna("") + " " + df["surname"].fillna("")
    for column in ("year", "grid", "finish_position", "points", "laps"):
        df[column] = numeric_values(df[column])
    df["points_known"] = df["points"].notna()
    # Exclusões de escopo são contadas no relatório, não confundidas com abandono.
    valid = df["year"].between(start_year, end_year) & df["grid"].gt(0)
    df = df.loc[valid].sort_values(["year", "race_id", "finish_position"]).reset_index(drop=True)
    report["joins"] = joins
    report["selection"]["output_results"] = int(len(df))
    df.attrs["data_quality"] = report
    return df
