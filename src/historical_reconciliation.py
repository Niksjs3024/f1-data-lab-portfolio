"""Reconciliação local e não destrutiva dos campos de volta rápida 2013–2014.

As referências externas são evidência auditável, não uma substituição do CSV raw.
Nenhuma função grava arquivos salvo `export_artifacts`, que exige destino novo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from .data_quality import parse_lap_seconds

YEARS = (2013, 2014)
AUDITED_FIELDS = ("fastestLapTime", "fastestLapSpeed", "fastestLap", "rank", "laps", "statusId")
REFERENCE_FIELDS = frozenset({"fastestLapTime", "fastestLapSpeed", "fastestLap", "rank", "laps", "statusId"})
STATUSES = frozenset({"confirmado", "divergente", "nao_verificado", "ambiguo"})
EVENT_KEY = ("year", "event_key", "circuit_id", "driver_id")
PAIR_KEY = ("driver_id", "constructor_id", "circuit_id")
RAW_FIELDS = {
    "fastestLapTime": "fastestLapTime",
    "fastestLapSpeed": "fastestLapSpeed",
    "fastestLap": "fastestLap",
    "rank": "rank",
    "laps": "laps",
    "statusId": "statusId",
}


def normalize_event(value: object) -> str:
    """Canonicaliza apenas o nome do evento; IDs externos nunca entram na chave."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def extract_raw_records(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Junta dados raw em cópias, sem recorte de grid e sem mutar as tabelas."""
    required = {"results", "races", "drivers", "constructors", "status"}
    missing = required.difference(tables)
    if missing:
        raise ValueError(f"Tabelas ausentes: {', '.join(sorted(missing))}")
    results = tables["results"].copy(deep=True)
    for name in ("fastestLapTime", "fastestLapSpeed", "fastestLap", "rank", "laps", "statusId"):
        if name not in results:
            results[name] = pd.NA
    if "grid" not in results:
        results["grid"] = pd.NA
    if results["resultId"].isna().any() or results["resultId"].duplicated().any():
        raise ValueError("resultId ausente ou duplicado; cardinalidade raw não é inequívoca")
    races = tables["races"].copy(deep=True)
    drivers = tables["drivers"].copy(deep=True)
    constructors = tables["constructors"].copy(deep=True)
    statuses = tables["status"].copy(deep=True)
    for frame, key in ((races, "raceId"), (drivers, "driverId"), (constructors, "constructorId"), (statuses, "statusId")):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Chave {key} ausente ou duplicada na dimensão raw")
    frame = results.merge(races[["raceId", "year", "circuitId", "name"]], on="raceId", how="left", validate="many_to_one")
    frame = frame.merge(drivers[["driverId", "forename", "surname"]], on="driverId", how="left", validate="many_to_one")
    frame = frame.merge(constructors[["constructorId", "name"]].rename(columns={"name": "constructor"}), on="constructorId", how="left", validate="many_to_one")
    frame = frame.merge(statuses[["statusId", "status"]], on="statusId", how="left", validate="many_to_one", suffixes=("", "_label"))
    frame = frame.rename(columns={"resultId": "result_id", "raceId": "race_id", "driverId": "driver_id", "constructorId": "constructor_id", "circuitId": "circuit_id", "name": "grand_prix"})
    frame["driver"] = frame["forename"].fillna("").astype(str).str.cat(frame["surname"].fillna("").astype(str), sep=" ").str.strip()
    for column in ("year", "circuit_id", "driver_id", "constructor_id", "statusId", "laps", "fastestLap", "rank", "grid"):
        frame[column] = _numeric(frame[column])
    frame = frame.loc[frame["year"].isin(YEARS)].copy()
    frame["event_key"] = frame["grand_prix"].map(normalize_event)
    if frame["result_id"].duplicated().any():
        raise ValueError("Junções multiplicaram resultId; abortando reconciliação")
    return frame.reset_index(drop=True)


def load_references(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Carrega uma tabela curada e valida campos/chaves sem usar raceId de provedor."""
    refs = source.copy(deep=True) if isinstance(source, pd.DataFrame) else pd.read_csv(source, keep_default_na=False)
    expected = {"year", "grand_prix", "circuit_id", "driver_id", "verified_fields", "source", "source_url", "consulted_at_utc", "confidence", "justification"}
    absent = expected.difference(refs.columns)
    if absent:
        raise ValueError(f"Colunas de referência ausentes: {', '.join(sorted(absent))}")
    refs["year"] = _numeric(refs["year"])
    refs["circuit_id"] = _numeric(refs["circuit_id"])
    refs["driver_id"] = _numeric(refs["driver_id"])
    refs["event_key"] = refs["grand_prix"].map(normalize_event)
    parsed = []
    for _, row in refs.iterrows():
        fields = frozenset(item.strip() for item in str(row["verified_fields"]).split(";") if item.strip())
        unknown = fields.difference(REFERENCE_FIELDS)
        if unknown:
            raise ValueError(f"Campos sem suporte na referência: {', '.join(sorted(unknown))}")
        if fields and (not str(row["source_url"]).startswith("https://") or not str(row["source"]).strip()):
            raise ValueError("Referência verificada requer fonte e URL HTTPS")
        parsed.append(fields)
    refs["_fields"] = parsed
    return refs


def _reference_value(row: pd.Series, field: str):
    column = f"official_{field}"
    if column not in row.index:
        return None
    value = row[column]
    if value is None or pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return None
    if field == "fastestLapTime":
        # Evidência armazena segundos numéricos com precisão declarada na fonte.
        return float(value)
    if field in {"fastestLap", "rank", "laps", "statusId"}:
        return int(float(value))
    return float(value)


def _original_value(row: pd.Series, field: str):
    return row[RAW_FIELDS[field]]


def _same_value(field: str, original, verified) -> bool:
    if verified is None or pd.isna(original):
        return False
    if field == "fastestLapTime":
        parsed = parse_lap_seconds(original)
        return pd.notna(parsed) and round_half_up(parsed, 1) == round_half_up(verified, 1)
    if field == "fastestLapSpeed":
        return round(float(original), 3) == round(float(verified), 3)
    return float(original) == float(verified)


def round_half_up(value: float, places: int = 1) -> float:
    quantum = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _correction_id(key: tuple, field: str, old, new, source_url: str) -> str:
    payload = json.dumps([*key, field, str(old), str(new), source_url], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def reconcile_records(raw: pd.DataFrame, references: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna uma linha por raw resultId e log campo a campo; não muda inputs."""
    data = raw.copy(deep=True)
    refs = load_references(references)
    required = {"result_id", "year", "grand_prix", "event_key", "circuit_id", "driver_id", "constructor_id"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Campos raw ausentes: {', '.join(sorted(missing))}")
    if data["result_id"].isna().any() or data["result_id"].duplicated().any():
        raise ValueError("result_id ausente ou duplicado")
    data["event_key"] = data["grand_prix"].map(normalize_event)
    raw_counts = data.groupby(list(EVENT_KEY), dropna=False)["result_id"].transform("size")
    ref_counts = refs.groupby(list(EVENT_KEY), dropna=False)["driver_id"].transform("size")
    ref_by_key = {}
    for _, ref in refs.iterrows():
        key = tuple(ref[name] for name in EVENT_KEY)
        ref_by_key.setdefault(key, []).append(ref)
    fields_out = {field: [] for field in AUDITED_FIELDS}
    corrections = []
    record_statuses = []
    for idx, row in data.iterrows():
        key = tuple(row[name] for name in EVENT_KEY)
        candidates = ref_by_key.get(key, [])
        ambiguous = int(raw_counts.loc[idx]) != 1 or len(candidates) > 1 or any(pd.isna(item) for item in key)
        ref = candidates[0] if len(candidates) == 1 else None
        statuses = []
        for field in AUDITED_FIELDS:
            original = _original_value(row, field)
            verified = _reference_value(ref, field) if ref is not None and field in ref["_fields"] else None
            if ambiguous:
                status = "ambiguo"
            elif ref is None or field not in ref["_fields"]:
                status = "nao_verificado"
            else:
                status = "confirmado" if _same_value(field, original, verified) else "divergente"
            source_url = str(ref["source_url"]) if ref is not None and field in ref["_fields"] else ""
            reason = str(ref["justification"]) if ref is not None and field in ref["_fields"] else "Fonte externa não verificou este campo/piloto/evento."
            reconciled = verified if status in {"confirmado", "divergente"} else (parse_lap_seconds(original) if field == "fastestLapTime" else original)
            fields_out[field].append({"original": original, "verified": verified, "reconciled": reconciled, "status": status, "source_url": source_url, "reason": reason})
            statuses.append(status)
            if status == "divergente":
                corr_id = _correction_id(key, field, original, verified, source_url)
                corrections.append({"correction_id": corr_id, "result_id": row["result_id"], "race_id": row.get("race_id"), "year": row["year"], "grand_prix": row["grand_prix"], "circuit_id": row["circuit_id"], "driver_id": row["driver_id"], "constructor_id": row["constructor_id"], "field": field, "original_value": original, "verified_value": verified, "status": status, "applied": True, "source": ref["source"], "source_url": source_url, "justification": reason})
        if "ambiguo" in statuses:
            record_statuses.append("ambiguo")
        elif "divergente" in statuses:
            record_statuses.append("divergente")
        elif any(status == "confirmado" for status in statuses):
            record_statuses.append("confirmado")
        else:
            record_statuses.append("nao_verificado")
    for field, entries in fields_out.items():
        data[f"{field}_original"] = [item["original"] for item in entries]
        data[f"{field}_verified"] = [item["verified"] for item in entries]
        data[f"{field}_reconciled"] = [item["reconciled"] for item in entries]
        data[f"{field}_verification_status"] = [item["status"] for item in entries]
        data[f"{field}_source_url"] = [item["source_url"] for item in entries]
    data["record_verification_status"] = record_statuses
    data["early_lap_count_alert"] = data["fastestLap"].notna() & data["laps"].notna() & data["fastestLap"].gt(data["laps"])
    log = pd.DataFrame(corrections, columns=["correction_id", "result_id", "race_id", "year", "grand_prix", "circuit_id", "driver_id", "constructor_id", "field", "original_value", "verified_value", "status", "applied", "source", "source_url", "justification"])
    if not log.empty and log["correction_id"].duplicated().any():
        raise ValueError("Correction ID collision; no export should be written")
    return data, log


def _stats(values: pd.DataFrame) -> dict:
    x = pd.to_numeric(values.get("difference_s", pd.Series(dtype=float)), errors="coerce").dropna()
    return {"pairs": int(len(x)), "mean_s": float(x.mean()) if len(x) else None, "median_s": float(x.median()) if len(x) else None, "positive": int(x.gt(0).sum()), "negative": int(x.lt(0).sum()), "zero": int(x.eq(0).sum())}


def _pair_table(records: pd.DataFrame) -> pd.DataFrame:
    data = records.copy(deep=True)
    data["original_seconds"] = data["fastestLapTime_original"].map(parse_lap_seconds)
    data["verified_seconds"] = pd.to_numeric(data["fastestLapTime_verified"], errors="coerce")
    data["reconciled_seconds"] = data["fastestLapTime_reconciled"].map(lambda v: float(v) if pd.notna(v) and isinstance(v, (int, float)) else parse_lap_seconds(v))
    data["time_verified"] = data["fastestLapTime_verification_status"].isin(["confirmado", "divergente"]) & data["verified_seconds"].notna()
    valid = data[data["year"].isin(YEARS)].copy()
    rows = []
    for key, group in valid.groupby(list(PAIR_KEY), dropna=False):
        by_year = {int(y): sub for y, sub in group.groupby("year")}
        if not all(y in by_year for y in YEARS):
            continue
        a, b = by_year[2013], by_year[2014]
        if len(a) != 1 or len(b) != 1:
            continue
        r13, r14 = a.iloc[0], b.iloc[0]
        if pd.isna(r13.original_seconds) or pd.isna(r14.original_seconds):
            continue
        time_both_verified = bool(r13.time_verified and r14.time_verified)
        # Scenario A is explicitly one-decimal: verified source values replace fields,
        # pending fields retain the unmodified raw value and remain visibly pending.
        a13 = round_half_up(r13.reconciled_seconds if r13.time_verified else r13.original_seconds, 1)
        a14 = round_half_up(r14.reconciled_seconds if r14.time_verified else r14.original_seconds, 1)
        # B is a strict verified subset; no fallback to raw or mixed precision.
        b13 = r13.verified_seconds if time_both_verified else None
        b14 = r14.verified_seconds if time_both_verified else None
        early_flags = []
        for item in (r13, r14):
            laps = pd.to_numeric(pd.Series([item.get("laps_original")]), errors="coerce").iloc[0]
            status = item.get("status")
            early_flags.append(bool(pd.notna(laps) and laps <= 3 and str(status).strip().casefold() != "finished"))
        rows.append({"driver_id": key[0], "constructor_id": key[1], "circuit_id": key[2], "grand_prix": r14.get("grand_prix"), "result_id_2013": r13.result_id, "result_id_2014": r14.result_id, "original_2013_s": float(r13.original_seconds), "original_2014_s": float(r14.original_seconds), "original_difference_s": float(r14.original_seconds-r13.original_seconds), "scenario_a_2013_s": a13, "scenario_a_2014_s": a14, "scenario_a_difference_s": float(a14-a13), "scenario_a_time_pairs_verified": int(r13.time_verified)+int(r14.time_verified), "scenario_b_eligible": time_both_verified, "scenario_b_2013_s": b13, "scenario_b_2014_s": b14, "scenario_b_difference_s": float(b14-b13) if time_both_verified else None, "early_retirement_proxy": any(early_flags), "laps_2013": r13.get("laps_original"), "laps_2014": r14.get("laps_original"), "status_2013": r13.get("status"), "status_2014": r14.get("status")})
    return pd.DataFrame(rows)


def _summary_block(frame: pd.DataFrame, difference_column: str) -> dict:
    valid = frame[pd.to_numeric(frame[difference_column], errors="coerce").notna()].copy()
    base = _stats(valid.rename(columns={difference_column: "difference_s"}))
    base["by_gp"] = {}
    for gp, group in valid.groupby("grand_prix", dropna=False):
        base["by_gp"][str(gp)] = _stats(group.rename(columns={difference_column: "difference_s"}))
    base["leave_one_gp_out"] = {}
    for gp in valid["grand_prix"].dropna().unique():
        rest = valid.loc[valid["grand_prix"].ne(gp)]
        base["leave_one_gp_out"][str(gp)] = _stats(rest.rename(columns={difference_column: "difference_s"}))
    japan = valid[valid["grand_prix"].astype(str).str.casefold().str.contains("japan|japanese")]
    base["japan_included"] = _stats(valid.rename(columns={difference_column: "difference_s"}))
    base["japan_excluded"] = _stats(valid.loc[~valid.index.isin(japan.index)].rename(columns={difference_column: "difference_s"}))
    early = valid[valid["early_retirement_proxy"]]
    base["early_retirement_proxy_definition"] = "qualquer uma das duas linhas tem laps <= 3 e status diferente de Finished; proxy de abandono precoce, não regra de exclusão"
    base["early_retirement_proxy_pairs"] = _stats(early.rename(columns={difference_column: "difference_s"}))
    base["excluding_early_retirement_proxy"] = _stats(valid.loc[~valid["early_retirement_proxy"]].rename(columns={difference_column: "difference_s"}))
    return base


def build_comparisons(records: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Constrói pares, comparações de mesma população e resumos auditáveis."""
    pairs = _pair_table(records)
    a = pairs.copy()
    b = pairs[pairs["scenario_b_eligible"]].copy()
    summaries = {
        "original_all_original_eligible_pairs": _summary_block(a.assign(difference_s=a.original_difference_s), "difference_s"),
        "scenario_a_all_original_eligible_pairs": _summary_block(a.assign(difference_s=a.scenario_a_difference_s), "difference_s"),
        "original_on_scenario_b_eligible_pairs": _summary_block(b.assign(difference_s=b.original_difference_s), "difference_s"),
        "scenario_b_on_same_scenario_b_eligible_pairs": _summary_block(b.assign(difference_s=b.scenario_b_difference_s), "difference_s"),
        "coverage": {"original_eligible_pairs": int(len(a)), "scenario_a_pairs_same_population": int(len(a)), "scenario_a_pairs_with_both_years_verified": int(a["scenario_a_time_pairs_verified"].eq(2).sum()), "scenario_b_fully_verified_pairs": int(len(b)), "scenario_b_fraction_of_original_pairs": float(len(b)/len(a)) if len(a) else 0.0, "same_pair_set_comparisons": ["original vs scenario A on original-eligible pair keys", "original vs scenario B on scenario-B-eligible pair keys"]},
        "interpretation": "Cenário A mantém um décimo de segundo e usa original nos campos temporais não verificados, sempre com status explícito. Cenário B contém somente pares com tempo verificado nos dois anos e precisão original da fonte; sua comparação com original usa exatamente as mesmas chaves.",
    }
    def delta(left, right):
        return left - right if left is not None and right is not None else None
    summaries["metric_differences_same_pairs"] = {"scenario_a_minus_original_mean_s": delta(summaries["scenario_a_all_original_eligible_pairs"]["mean_s"], summaries["original_all_original_eligible_pairs"]["mean_s"]), "scenario_a_minus_original_median_s": delta(summaries["scenario_a_all_original_eligible_pairs"]["median_s"], summaries["original_all_original_eligible_pairs"]["median_s"]), "scenario_b_minus_original_common_mean_s": delta(summaries["scenario_b_on_same_scenario_b_eligible_pairs"]["mean_s"], summaries["original_on_scenario_b_eligible_pairs"]["mean_s"]), "scenario_b_minus_original_common_median_s": delta(summaries["scenario_b_on_same_scenario_b_eligible_pairs"]["median_s"], summaries["original_on_scenario_b_eligible_pairs"]["median_s"])}
    return pairs, summaries


def coverage_report(records: pd.DataFrame, references: pd.DataFrame, corrections: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    refs = load_references(references)
    fields = {}
    for field in AUDITED_FIELDS:
        state_column = f"{field}_verification_status"
        fields[field] = {str(status): int(records[state_column].eq(status).sum()) for status in sorted(STATUSES)}
    return {"source_record_count_2013_2014": int(len(records)), "raw_event_count": int(records[["year", "circuit_id", "event_key"]].drop_duplicates().shape[0]), "referenced_event_driver_rows": int(len(refs)), "reference_distinct_events": int(refs[list(("year", "event_key", "circuit_id"))].drop_duplicates().shape[0]), "reference_fields_claimed": {field: int(refs["_fields"].map(lambda fields: field in fields).sum()) for field in AUDITED_FIELDS}, "field_status_counts_over_raw_records": fields, "correction_fields": corrections["field"].value_counts().to_dict() if not corrections.empty else {}, "corrected_result_rows": int(corrections["result_id"].nunique()) if not corrections.empty else 0, "ambiguous_rows": int(records["record_verification_status"].eq("ambiguo").sum()), "internal_fastest_lap_gt_completed_laps_alerts": int(records["early_lap_count_alert"].sum()), "original_eligible_pair_count": int(len(pairs)), "official_precision_pair_count": int(pairs["scenario_b_eligible"].sum())}


def export_artifacts(raw_records: pd.DataFrame, reconciled_records: pd.DataFrame, corrections: pd.DataFrame, pairs: pd.DataFrame, summaries: dict, coverage: dict, output_dir: str | Path) -> Path:
    """Exporta apenas em diretório ainda inexistente; recusa substituir qualquer artefato."""
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"Destino já existe; nada foi sobrescrito: {target}")
    target.mkdir(parents=True, exist_ok=False)
    try:
        raw_records.to_csv(target / "original_records_2013_2014.csv", index=False, lineterminator="\n")
        reconciled_records.to_csv(target / "reconciled_records_2013_2014.csv", index=False, lineterminator="\n")
        corrections.to_csv(target / "field_corrections_2013_2014.csv", index=False, lineterminator="\n")
        pairs.to_csv(target / "paired_scenarios_2013_2014.csv", index=False, lineterminator="\n")
        bundle = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "coverage": coverage, "comparisons": summaries, "sources": {"reference_table": "data/reconciliation/verified_fastest_laps_2013_2014.csv (schema only; no official values transcribed pending reuse authorization)", "official_pages_cited_in_prior_audit": ["https://www.formula1.com/en/results/2013/races/879/australia/fastest-laps", "https://www.formula1.com/en/results/2014/races/898/australia/fastest-laps", "https://www.formula1.com/en/results/2014/races/903/monaco/fastest-laps", "https://www.formula1.com/en/results/2014/races/908/hungary/fastest-laps", "https://www.formula1.com/en/results/2014/races/912/japan/fastest-laps"], "terms_check": "A página da Austrália 2014 foi consultada manualmente em 2026-10-22; contém copyright e não exibe licença ou autorização de reutilização. Nenhuma transcrição foi incluída."}, "climate_schema_plan": ["year", "grand_prix", "race_date", "circuit_id", "race_conditions", "track_condition_at_fastest_lap", "rain", "air_temperature", "track_temperature", "track_changes", "interruptions", "source", "source_url", "granularity", "certainty", "matching_method", "reuse_terms"]}
        (target / "reconciliation_summary_2013_2014.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (target / "reconciliation_summary_2013_2014.md").write_text(render_summary_markdown(bundle), encoding="utf-8")
    except Exception:
        # Best-effort transactional cleanup is limited to the newly created target.
        import shutil
        shutil.rmtree(target, ignore_errors=True)
        raise
    return target


def render_summary_markdown(bundle: dict) -> str:
    comparison = bundle["comparisons"]
    coverage = bundle["coverage"]
    lines = ["# Fase 2 — Reconciliação auditável 2013–2014", "", f"Gerado (UTC): {bundle['generated_at_utc']}", "", "**Uso interno de auditoria.** Não é homologação dos indicadores históricos nem validação externa integral. Nenhuma fonte oficial foi confirmada quanto à licença de redistribuição.", "", "## Cobertura", "", "```json", json.dumps(coverage, ensure_ascii=False, indent=2), "```", "", "## Comparações e mesma população", "", "Cada resumo informa n e métricas. A comparação A usa 0,1 s e conserva valores originais apenas nos campos pendentes, com status presente no CSV reconciliado. B usa somente pares com ambos os anos verificados; o comparativo original/B usa exatamente essas mesmas chaves.", "", "```json", json.dumps(comparison, ensure_ascii=False, indent=2), "```", "", "## Limites e preparação climática", "", "`laps` e `statusId` não são verificados pela página de voltas rápidas. Alertas internos de plausibilidade não estabelecem identidade. Não se infere clima a partir dos tempos. Schema climático proposto: " + ", ".join(bundle["climate_schema_plan"]) + ". A condição no instante da volta só deve ser preenchida com fonte temporalmente granular e identificável; registrar fonte, URL, granularidade, certeza, método de vínculo e termos de reutilização.", ""]
    return "\n".join(lines)
