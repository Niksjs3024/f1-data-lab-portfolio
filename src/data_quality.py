"""Contratos de ingestão e auditoria não destrutiva dos CSVs históricos."""
import math

import numpy as np
import pandas as pd

REQUIRED_FILES = ("results", "races", "drivers", "constructors", "status")
PRIMARY_KEYS = {"results": "resultId", "races": "raceId", "drivers": "driverId",
                "constructors": "constructorId", "status": "statusId"}
FOREIGN_KEYS = {"raceId": "races", "driverId": "drivers",
                "constructorId": "constructors", "statusId": "status"}
REQUIRED_COLUMNS = {
    "results": ["resultId", "raceId", "driverId", "constructorId", "statusId",
                "grid", "positionOrder", "points"],
    "races": ["raceId", "year", "circuitId", "name"],
    "drivers": ["driverId", "forename", "surname"],
    "constructors": ["constructorId", "name"],
    "status": ["statusId", "status"],
}
OPTIONAL_RESULT_COLUMNS = ("laps", "position", "positionText", "fastestLapTime", "fastestLapSpeed")


class DataValidationError(ValueError):
    """Falha de contrato; o relatório completo continua disponível em report."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def missing_mask(values: pd.Series) -> pd.Series:
    """Reconhece sentinelas sem converter pontuação ausente em zero."""
    text = values.astype("string").str.strip()
    return (values.isna() | text.isin(["", "\\N", "N/A", "NA", "NULL", "null", "\x00"])).fillna(True)


def numeric_values(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.mask(missing_mask(values)), errors="coerce")


def invalid_numeric(values: pd.Series, *, minimum: float = 0,
                    integer: bool = False, allow_missing: bool = False) -> pd.Series:
    numbers = numeric_values(values)
    finite = pd.Series(np.isfinite(numbers.to_numpy(dtype=float, na_value=np.nan)), index=values.index)
    invalid = ~finite | numbers.lt(minimum).fillna(False)
    if integer:
        invalid |= numbers.where(finite).mod(1).ne(0).fillna(False)
    if allow_missing:
        invalid &= ~missing_mask(values)
    return invalid.fillna(True)


def parse_lap_seconds(value: object) -> float:
    """Aceita M:SS.s ou H:MM:SS.s positivos, finitos e com componentes válidos."""
    try:
        if pd.isna(value):
            return float("nan")
        tokens = str(value).strip().split(":")
        if len(tokens) not in (2, 3):
            return float("nan")
        parts = [float(token) for token in tokens]
        if not all(math.isfinite(part) and part >= 0 for part in parts):
            return float("nan")
        if any(not token.strip().isdigit() for token in tokens[:-1]):
            return float("nan")
        if parts[-1] >= 60 or (len(parts) == 3 and parts[1] >= 60):
            return float("nan")
        seconds = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
        return seconds if seconds > 0 and math.isfinite(seconds) else float("nan")
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def json_safe(value):
    """Normaliza escalares para JSON estrito sem confundir infinito com ausência."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _records(frame: pd.DataFrame, limit: int = 8) -> list[dict]:
    return json_safe(frame.head(limit).to_dict(orient="records"))


def _range(values: pd.Series) -> dict:
    valid = numeric_values(values).dropna()
    valid = valid[np.isfinite(valid)]
    return {"min": int(valid.min()) if len(valid) else None,
            "max": int(valid.max()) if len(valid) else None}


def audit_tables(tables: dict[str, pd.DataFrame], start_year: int = 2004,
                 end_year: int = 2017) -> dict:
    """Relata anomalias sem lançar erro, excluir, imputar ou corrigir entradas.

    blocking distingue corrupção estrutural/dos cálculos de avisos históricos.
    count sempre conta linhas afetadas, salvo quando a mensagem indicar grupos.
    """
    report = {"schema_version": 1, "scope": {"start_year": start_year, "end_year": end_year},
              "tables": {}, "issues": [], "coverage": {}, "selection": {},
              "foreign_keys": {}, "numeric_checks": {}}

    def issue(code, table, mask=None, *, message, blocking=False, severity="warning", frame=None):
        source = tables.get(table, pd.DataFrame()) if frame is None else frame
        affected = source if mask is None else source.loc[mask]
        report["issues"].append({"code": code, "table": table,
                                 "severity": "error" if blocking else severity,
                                 "blocking": blocking, "count": int(len(affected)),
                                 "message": message, "examples": _records(affected)})

    if not isinstance(start_year, int) or not isinstance(end_year, int) or start_year > end_year:
        issue("invalid_year_range", "races", message="Intervalo de anos inválido.", blocking=True)
        return report
    for name in REQUIRED_FILES:
        if name not in tables:
            issue("missing_table", name, message=f"Tabela obrigatória ausente: {name}.", blocking=True)
            continue
        table = tables[name]
        absent = sorted(set(REQUIRED_COLUMNS[name]) - set(table.columns))
        report["tables"][name] = {
            "rows": int(len(table)), "columns": list(table.columns),
            "missing_by_column": {col: int(missing_mask(table[col]).sum()) for col in table},
            "duplicate_rows": int(table.duplicated(keep=False).sum()),
            "primary_key": PRIMARY_KEYS[name],
        }
        if absent:
            issue("missing_columns", name, message=f"Colunas obrigatórias ausentes: {', '.join(absent)}.", blocking=True)
        key = PRIMARY_KEYS[name]
        if key in table:
            bad = invalid_numeric(table[key], minimum=1, integer=True)
            report["tables"][name]["invalid_primary_key_rows"] = int(bad.sum())
            if bad.any():
                issue("invalid_primary_key", name, bad, message=f"Chave {key} ausente ou inválida.", blocking=True)
            duplicate = numeric_values(table[key]).duplicated(keep=False) & ~missing_mask(table[key])
            report["tables"][name]["duplicate_primary_key_rows"] = int(duplicate.sum())
            if duplicate.any():
                issue("duplicate_primary_key", name, duplicate, message=f"Chave {key} repetida.", blocking=True)
        if table.duplicated(keep=False).any():
            issue("duplicate_rows", name, table.duplicated(keep=False), message="Linhas integralmente repetidas; nenhuma removida.")
        if any(str(col).startswith("Unnamed:") for col in table):
            issue("unnamed_column", name, message="Coluna sem nome no CSV; conferir cabeçalho, sem reescrever original.")

    schema_ok = all(name in tables and set(cols).issubset(tables[name].columns)
                    for name, cols in REQUIRED_COLUMNS.items())
    if not schema_ok:
        return report

    results, races = tables["results"], tables["races"]
    for key, parent in FOREIGN_KEYS.items():
        child_values = numeric_values(results[key])
        parent_values = numeric_values(tables[parent][key])
        bad = invalid_numeric(results[key], minimum=1, integer=True) | ~child_values.isin(parent_values)
        report["foreign_keys"][key] = {"parent": parent, "checked_rows": int(len(results)),
                                      "invalid_or_unmatched_rows": int(bad.sum())}
        if bad.any():
            issue("foreign_key", "results", bad, message=f"Referência {key} ausente, inválida ou sem correspondência em {parent}.", blocking=True)

    for col in ("year", "circuitId"):
        bad = invalid_numeric(races[col], minimum=1, integer=True)
        if bad.any():
            issue("invalid_race_field", "races", bad, message=f"{col} ausente ou inválido impede determinar contexto.", blocking=True)

    # Mapeamento não multiplica linhas mesmo quando a dimensão está inválida.
    race_index = pd.DataFrame({"id": numeric_values(races["raceId"]), "year": numeric_values(races["year"])})
    race_years = race_index.drop_duplicates("id").set_index("id")["year"]
    years = numeric_values(results["raceId"]).map(race_years)
    in_period = years.between(start_year, end_year)
    grid = numeric_values(results["grid"])
    finish = numeric_values(results["positionOrder"])
    eligible = in_period & grid.gt(0) & ~invalid_numeric(results["grid"], integer=True) & ~invalid_numeric(results["positionOrder"], minimum=1, integer=True)
    report["coverage"] = {
        "calendar_years": _range(races["year"]), "result_years": _range(years),
        "calendar_by_year": {str(int(year)): int(count) for year, count in numeric_values(races["year"]).value_counts().sort_index().items() if pd.notna(year) and np.isfinite(year)},
        "results_by_year": {str(int(year)): int(count) for year, count in years.value_counts().sort_index().items() if pd.notna(year) and np.isfinite(year)},
        "races_without_results": _records(races.loc[~numeric_values(races["raceId"]).isin(numeric_values(results["raceId"]))], limit=len(races)),
    }
    report["selection"] = {
        "input_results": int(len(results)), "outside_period": int((~in_period).sum()),
        "within_period": int(in_period.sum()), "grid_zero_excluded": int((in_period & grid.eq(0)).sum()),
        "invalid_grid_rows": int((in_period & invalid_numeric(results["grid"], integer=True)).sum()),
        "invalid_finish_rows": int((in_period & invalid_numeric(results["positionOrder"], minimum=1, integer=True)).sum()),
        "eligible_results": int(eligible.sum()),
        "eligible_races": int(numeric_values(results.loc[eligible, "raceId"]).nunique()),
        "unknown_points": int((eligible & missing_mask(results["points"])).sum()),
    }
    zero_grid = in_period & grid.eq(0)
    if zero_grid.any():
        issue("grid_zero_excluded", "results", zero_grid, severity="info",
              message="Grid zero excluído apenas por falta de posição comparável; não recodificado como último grid.")
    # Cobertura das dimensões significa corridas/anos associados, não data de nascimento.
    for name, key in PRIMARY_KEYS.items():
        if name == "races":
            associated = numeric_values(races["year"])
        elif name == "results":
            associated = years
        else:
            associated = years.loc[numeric_values(results[key]).isin(numeric_values(tables[name][key]))]
        report["tables"][name]["associated_result_years"] = _range(associated)
    report["tables"]["races"]["calendar_years"] = _range(races["year"])

    for col, minimum, integer, allow_missing in (("grid", 0, True, False), ("positionOrder", 1, True, False), ("points", 0, False, True)):
        bad = invalid_numeric(results[col], minimum=minimum, integer=integer, allow_missing=allow_missing)
        report["numeric_checks"][col] = {"invalid_rows": int(bad.sum()),
                                          "invalid_rows_in_period": int((bad & in_period).sum()),
                                          "missing_rows": int(missing_mask(results[col]).sum())}
        if (bad & in_period).any():
            issue("invalid_numeric", "results", bad & in_period, message=f"{col}: valor inválido no período selecionado.", blocking=True)
        if (bad & ~in_period).any():
            issue("invalid_numeric_outside_period", "results", bad & ~in_period, message=f"{col}: valor inválido fora do período; não afeta os cálculos selecionados.")
    unknown_points = missing_mask(results["points"]) & in_period
    if unknown_points.any():
        issue("unknown_points", "results", unknown_points, message="Pontos desconhecidos preservados: não significam zero; taxa de pontuação do grid fica indisponível.")

    participation = pd.DataFrame({col: numeric_values(results[col]) for col in ("raceId", "driverId")})
    repeated = participation.duplicated(["raceId", "driverId"], keep=False)
    modern = in_period & years.ge(2004)
    report["participations"] = {"duplicate_rows": int(repeated.sum()),
                                  "duplicate_groups": int(participation.loc[repeated].drop_duplicates().shape[0]),
                                  "duplicate_modern_rows_in_period": int((repeated & modern).sum())}
    if (repeated & modern).any():
        issue("duplicate_participation", "results", repeated & modern, message="Participação corrida/piloto repetida no recorte moderno.", blocking=True)
    if (repeated & ~modern).any():
        issue("historical_participation", "results", repeated & ~modern, message="Participações repetidas históricas/fora do recorte; exigem contexto, sem deduplicação automática.")

    positive = results.loc[eligible].copy()
    keys = pd.DataFrame({"raceId": numeric_values(positive["raceId"]), "grid": numeric_values(positive["grid"])})
    repeated_grid = keys.duplicated(["raceId", "grid"], keep=False)
    report["selection"]["duplicate_grid_groups"] = _records(keys.loc[repeated_grid].drop_duplicates(), limit=len(keys))
    if repeated_grid.any():
        issue("duplicate_grid", "results", repeated_grid, frame=positive,
              message="Grid positivo repetido na mesma corrida; mantido para compatibilidade. Não trocar apenas o denominador nem escolher um piloto arbitrariamente.")
    report["selection"]["grid_counts"] = [
        {"grid": int(pos), "records": int(len(group)), "distinct_races": int(numeric_values(group["raceId"]).nunique())}
        for pos, group in positive.groupby(numeric_values(positive["grid"]), sort=True)
    ]
    if "fastestLapTime" in results:
        report["selection"]["missing_fastest_lap_preserved"] = int((eligible & missing_mask(results["fastestLapTime"])).sum())
        bad_lap = ~missing_mask(results["fastestLapTime"]) & results["fastestLapTime"].map(parse_lap_seconds).isna()
        if bad_lap.any():
            issue("invalid_fastest_lap", "results", bad_lap, message="Volta rápida inválida: resultado preservado; não utilizada no pareamento de voltas.")
    for col in OPTIONAL_RESULT_COLUMNS:
        if col not in results:
            issue("optional_column_absent", "results", message=f"Campo opcional {col} ausente; preservado como desconhecido.", severity="info")
    if "laps" in results:
        missing_laps = missing_mask(results["laps"]) & in_period
        if missing_laps.any():
            issue("missing_laps", "results", missing_laps, message="Voltas desconhecidas preservadas; ausência não equivale a zero voltas.")
    for col, minimum, integer in (("laps", 0, True), ("position", 1, True), ("fastestLapSpeed", 0, False)):
        if col in results:
            bad = invalid_numeric(results[col], minimum=minimum, integer=integer, allow_missing=True)
            if bad.any():
                issue("invalid_optional_numeric", "results", bad, message=f"{col} inválido; preservado para investigação.", blocking=col == "laps" and bool((bad & in_period).any()))
    if "laps" in results:
        status_normalized = tables["status"].copy()
        status_normalized["statusId"] = numeric_values(status_normalized["statusId"])
        status_names = status_normalized.drop_duplicates("statusId").set_index("statusId")["status"]
        withdrawn = (eligible & numeric_values(results["statusId"]).map(status_names).eq("Withdrew") & numeric_values(results["laps"]).eq(0))
        if withdrawn.any():
            issue("withdrawn_zero_laps", "results", withdrawn, message="Retirada com grid positivo e zero voltas: grid atribuído não comprova largada; nenhuma exclusão automática.")
    drivers = tables["drivers"]
    suspicious = (drivers["forename"].astype("string").str.contains("Ì|�", regex=True, na=False)
                  | drivers["surname"].astype("string").str.contains("Ì|�", regex=True, na=False))
    if suspicious.any():
        issue("suspected_name_encoding", "drivers", suspicious, message="Marcadores suspeitos de codificação; grafia não corrigida sem fonte confirmada.")
    if "date" in races and "round" in races:
        ordered = races.assign(_year=numeric_values(races["year"]), _round=numeric_values(races["round"]),
                               _date=pd.to_datetime(races["date"], errors="coerce")).sort_values(["_year", "_round"])
        inversion = ordered.groupby("_year")["_date"].diff().dt.days.lt(0)
        if inversion.any():
            issue("calendar_date_order", "races", inversion, frame=ordered.drop(columns=["_year", "_round", "_date"]),
                  message="Data anterior à rodada precedente; investigar também a etapa precedente, sem atribuir data corrigida.")
    return report


def enforce_report(report: dict, *, strict: bool = False) -> None:
    rejected = [item for item in report["issues"] if item["blocking"] or (strict and item["severity"] == "warning")]
    if rejected:
        codes = ", ".join(dict.fromkeys(item["code"] for item in rejected))
        raise DataValidationError(f"Validação dos dados falhou: {codes}. Consulte o relatório de auditoria.", report)


def validate_tables(tables: dict[str, pd.DataFrame], start_year: int = 2004,
                    end_year: int = 2017, *, strict: bool = False) -> dict:
    report = audit_tables(tables, start_year, end_year)
    enforce_report(report, strict=strict)
    return report


def validate_analysis_frame(df: pd.DataFrame) -> None:
    """Protege também chamadas diretas às probabilidades com dados sintéticos."""
    required = {"result_id", "grid", "finish_position", "points"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colunas analíticas ausentes: {', '.join(sorted(required - set(df.columns)))}")
    for col, minimum, integer, nullable in (("result_id", 1, True, False), ("grid", 1, True, False),
                                          ("finish_position", 1, True, False), ("points", 0, False, True)):
        if invalid_numeric(df[col], minimum=minimum, integer=integer, allow_missing=nullable).any():
            raise ValueError(f"Campo analítico inválido: {col}.")
    if numeric_values(df["result_id"]).duplicated().any():
        raise ValueError("result_id duplicado nos indicadores.")
