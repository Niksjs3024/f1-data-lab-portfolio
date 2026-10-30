"""Independent NOAA/event climate data preparation utilities.

This module deliberately has no dependency on the dashboard, historical lap
analysis, or model code. NOAA values retain the source tokens and quality flags.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

NOAA_ENDPOINT = "https://www.ncei.noaa.gov/access/services/data/v1"
WINDOW_BEFORE = timedelta(hours=2)
WINDOW_AFTER = timedelta(hours=4)
MISSING_ISD_VALUE = 9999
TEMP_TOKEN = re.compile(r"^[+-]?\d{4}$")
ISD_TEMP_DEW_QUALITY_SOURCE = (
    "https://www.ncei.noaa.gov/pub/data/noaa/isd-format-document.pdf"
)
ISD_TEMP_DEW_QUALITY_PAGE = 11
ISD_TEMP_DEW_QUALITY_POLICY = "isd-tmp-dew-qc-p11-v1"

# NOAA ISD format document, p. 11. This map applies only to TMP and DEW.
ISD_TEMP_DEW_QUALITY_CODES = {
    "0": ("gross_limits_only", "Passed gross limits check", False),
    "1": ("passed_all_qc", "Passed all quality control checks", True),
    "2": ("suspect", "Suspect", False),
    "3": ("erroneous", "Erroneous", False),
    "4": ("gross_limits_only_ncei", "Passed gross limits check, data originate from an NCEI data source", False),
    "5": ("passed_all_qc_ncei", "Passed all quality control checks, data originate from an NCEI data source", True),
    "6": ("suspect_ncei", "Suspect, data originate from an NCEI data source", False),
    "7": ("erroneous_ncei", "Erroneous, data originate from an NCEI data source", False),
    "9": ("gross_limits_if_present", "Passed gross limits check if element is present", False),
    "A": ("suspect_accepted_as_good", "Data value flagged as suspect, but accepted as a good value", False),
    "C": ("awos_whole_degree", "AWOS whole-degree value; automated QC flags these values, but they are accepted as valid", False),
    "I": ("inserted_by_validator", "Data value not originally in data, but inserted by validator", False),
    "M": ("manually_changed", "Manual changes made to value based on information provided by NWS or FAA", False),
    "P": ("replaced_by_validator", "Data value not originally flagged as suspect, but replaced by validator", False),
    "R": ("replaced_by_ncei_software", "Data value replaced with value computed by NCEI software", False),
    "U": ("replaced_with_edited_value", "Data value replaced with edited value", False),
}

EVENT_COLUMNS = (
    "season", "grand_prix", "race_id", "circuit_id", "circuit_name",
    "race_date_local", "scheduled_start_utc", "timezone",
    "scheduled_start_local", "utc_offset", "documented_start_local",
    "documented_start_source_url", "actual_start_utc",
    "actual_start_source_url", "circuit_latitude", "circuit_longitude",
    "circuit_source_url", "scheduled_time_source",
)
OBSERVATION_COLUMNS = (
    "season", "grand_prix", "race_id", "circuit_id", "scheduled_start_utc",
    "window_start_utc", "window_end_utc_exclusive", "in_window",
    "station_id", "station_name", "station_latitude", "station_longitude",
    "station_elevation_m", "distance_to_circuit_km", "observation_time_utc",
    "report_type", "tmp_raw", "tmp_quality_flag", "air_temperature_c",
    "tmp_missing_reason", "dew_raw", "dew_quality_flag", "dew_point_c",
    "dew_missing_reason", "source_url", "consulted_at_utc",
    "response_sha256", "quality_flags_interpreted",
    "duplicate_key_count_for_event_window",
)
QUERY_COLUMNS = (
    "season", "grand_prix", "race_id", "station_id", "window_start_utc",
    "window_end_utc_exclusive", "request_url", "consulted_at_utc",
    "response_sha256", "response_records", "window_records", "query_status",
    "error_message",
)


def parse_utc(value: str) -> datetime:
    """Parse a timestamp and return an aware UTC datetime; naive input is UTC."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def window_for_event(scheduled_start_utc: str | datetime) -> tuple[datetime, datetime]:
    """Return the fixed half-open observation window [T-2h, T+4h)."""
    start = (parse_utc(scheduled_start_utc) if isinstance(scheduled_start_utc, str)
             else scheduled_start_utc.astimezone(timezone.utc))
    return start - WINDOW_BEFORE, start + WINDOW_AFTER


def parse_isd_measurement(raw: Any) -> dict[str, Any]:
    """Decode an ISD temperature/dew-point token while preserving its flag.

    NOAA Global Hourly emits values such as ``+0290,1``. The signed four-digit
    value is tenths of a degree Celsius. The first attribute is retained as a
    raw quality flag; no flag is silently discarded or treated as a QC pass.
    """
    if raw is None or str(raw).strip() == "":
        return {"raw": "" if raw is None else str(raw), "value_c": None,
                "quality_flag": None, "missing_reason": "field_absent"}

    text = str(raw).strip()
    parts = text.split(",")
    value_token = parts[0].strip()
    quality_flag = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

    if not TEMP_TOKEN.fullmatch(value_token):
        return {"raw": text, "value_c": None, "quality_flag": quality_flag,
                "missing_reason": "malformed_value_token"}

    raw_integer = int(value_token)
    if abs(raw_integer) == MISSING_ISD_VALUE:
        return {"raw": text, "value_c": None, "quality_flag": quality_flag,
                "missing_reason": "isd_sentinel_9999"}

    value_c = raw_integer / 10.0
    if not math.isfinite(value_c):
        return {"raw": text, "value_c": None, "quality_flag": quality_flag,
                "missing_reason": "non_finite_value"}
    return {"raw": text, "value_c": value_c, "quality_flag": quality_flag,
            "missing_reason": ""}


def interpret_isd_quality_flag(flag: Any, field: str) -> dict[str, Any]:
    """Interpret only the NOAA ISD TMP/DEW attribute documented on page 11.

    The principal QC policy accepts codes 1 and 5, which passed all QC checks.
    Other documented codes retain their source classification but are not
    eligible for the QC-filtered statistics. Unknown or absent flags are not
    accepted.
    """
    normalized_field = str(field).strip().upper()
    if normalized_field not in {"TMP", "DEW"}:
        raise ValueError("quality flag interpretation is limited to TMP and DEW")
    code = "" if flag is None else str(flag).strip().upper()
    mapped = ISD_TEMP_DEW_QUALITY_CODES.get(code)
    if mapped is None:
        return {
            "quality_flag": code or None,
            "quality_class": "missing_flag" if not code else "unknown_code",
            "quality_description": "No quality flag supplied" if not code else "Unmapped quality code",
            "accepted_all_qc": False,
        }
    classification, description, passed_all_qc = mapped
    return {
        "quality_flag": code,
        "quality_class": classification,
        "quality_description": description,
        "accepted_all_qc": passed_all_qc,
    }


def classify_observation_quality(
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy observations and add independent TMP/DEW quality assessments."""
    classified: list[dict[str, Any]] = []
    for source_row in observations:
        row = dict(source_row)
        for field, value_column in (("TMP", "air_temperature_c"), ("DEW", "dew_point_c")):
            prefix = field.lower()
            interpreted = interpret_isd_quality_flag(row.get(f"{prefix}_quality_flag"), field)
            has_numeric_value = row.get(value_column) not in (None, "")
            row[f"{prefix}_quality_class"] = interpreted["quality_class"]
            row[f"{prefix}_quality_description"] = interpreted["quality_description"]
            row[f"{prefix}_qc_eligible"] = bool(
                has_numeric_value and interpreted["accepted_all_qc"]
            )
        row["quality_interpretation_applied"] = True
        row["quality_policy_version"] = ISD_TEMP_DEW_QUALITY_POLICY
        row["quality_policy_source"] = ISD_TEMP_DEW_QUALITY_SOURCE
        classified.append(row)
    return classified


def validate_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return event-clock/key issues without changing the supplied mappings."""
    issues: list[str] = []
    seen_races: set[str] = set()
    seen_event_keys: set[tuple[str, str]] = set()
    for row in events:
        race_id = str(row.get("race_id", "")).strip()
        event_key = (str(row.get("season", "")).strip(),
                     str(row.get("grand_prix", "")).strip())
        if not race_id or race_id in seen_races:
            issues.append(f"missing_or_duplicate_race_id:{race_id}")
        if event_key in seen_event_keys or not all(event_key):
            issues.append(f"missing_or_duplicate_event_key:{event_key}")
        seen_races.add(race_id)
        seen_event_keys.add(event_key)

        try:
            scheduled = parse_utc(str(row["scheduled_start_utc"]))
            local = datetime.fromisoformat(str(row["scheduled_start_local"]))
            zone = ZoneInfo(str(row["timezone"]))
            converted = scheduled.astimezone(zone)
            if local.utcoffset() != converted.utcoffset() or local.replace(tzinfo=None) != converted.replace(tzinfo=None):
                issues.append(f"scheduled_local_mismatch:{race_id}")
            expected_offset = converted.strftime("%z")
            supplied_offset = str(row.get("utc_offset", "")).replace(":", "")
            if supplied_offset and expected_offset != supplied_offset:
                issues.append(f"utc_offset_mismatch:{race_id}")
        except (KeyError, ValueError, TypeError):
            issues.append(f"invalid_event_clock:{race_id}")

        actual = str(row.get("actual_start_utc", "")).strip()
        actual_source = str(row.get("actual_start_source_url", "")).strip()
        if actual and not actual_source:
            issues.append(f"actual_start_without_source:{race_id}")
        if not actual and actual_source:
            issues.append(f"source_without_actual_start:{race_id}")
    return issues


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km using the mean Earth radius (6,371.0088 km)."""
    try:
        lat1, lon1, lat2, lon2 = (float(value) for value in (lat1, lon1, lat2, lon2))
    except (TypeError, ValueError) as exc:
        raise ValueError("coordinates must be numeric") from exc
    coords = (lat1, lon1, lat2, lon2)
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("coordinates must be finite")
    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        raise ValueError("latitude outside [-90, 90]")
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        raise ValueError("longitude outside [-180, 180]")
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 6371.0088 * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def records_in_window(records: Iterable[Mapping[str, Any]], scheduled_start_utc: str) -> list[Mapping[str, Any]]:
    """Keep only rows whose DATE falls inside the six-hour half-open window."""
    start, end = window_for_event(scheduled_start_utc)
    selected = []
    for row in records:
        observed = parse_utc(str(row["DATE"]))
        if start <= observed < end:
            selected.append(row)
    return sorted(selected, key=lambda row: (parse_utc(str(row["DATE"])),
                                             str(row.get("REPORT_TYPE", ""))))


def audit_observation_duplicates(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report duplicate timestamp/report-type keys without dropping any row."""
    counts: dict[tuple[str, str, str], int] = {}
    for row in records:
        key = (str(row.get("STATION", "")), str(row.get("DATE", "")),
               str(row.get("REPORT_TYPE", "")))
        counts[key] = counts.get(key, 0) + 1
    duplicate_keys = [key for key, count in counts.items() if count > 1]
    return {"duplicate_key_count": len(duplicate_keys),
            "duplicate_keys": duplicate_keys,
            "multiple_report_type_timestamps": _multiple_report_types(records)}


def _multiple_report_types(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[str]]]:
    report_types: dict[str, set[str]] = {}
    for row in records:
        timestamp = str(row.get("DATE", ""))
        report_types.setdefault(timestamp, set()).add(str(row.get("REPORT_TYPE", "")))
    return [(timestamp, sorted(types)) for timestamp, types in report_types.items() if len(types) > 1]


def _measurement_columns(raw: Any, prefix: str) -> dict[str, Any]:
    parsed = parse_isd_measurement(raw)
    return {
        f"{prefix.lower()}_raw": parsed["raw"],
        f"{prefix.lower()}_quality_flag": parsed["quality_flag"] or "",
        "air_temperature_c" if prefix == "TMP" else "dew_point_c": parsed["value_c"],
        f"{prefix.lower()}_missing_reason": parsed["missing_reason"],
    }


def observation_rows_for_event(
    event: Mapping[str, Any],
    station: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    source_url: str,
    consulted_at_utc: str,
    response_sha256: str,
) -> list[dict[str, Any]]:
    """Build normalized rows for the event window; preserve same-time reports."""
    schedule = str(event["scheduled_start_utc"])
    window_start, window_end = window_for_event(schedule)
    event_records = records_in_window(records, schedule)
    duplicates = audit_observation_duplicates(event_records)
    rows: list[dict[str, Any]] = []
    for record in event_records:
        station_id = str(record.get("STATION", "")).strip()
        if station_id != str(station["station_id"]):
            raise ValueError(f"station ID mismatch: expected {station['station_id']}, got {station_id}")
        distance = haversine_km(
            float(station["station_latitude"]), float(station["station_longitude"]),
            float(event["circuit_latitude"]), float(event["circuit_longitude"]),
        )
        rows.append({
            "season": event["season"], "grand_prix": event["grand_prix"],
            "race_id": event["race_id"], "circuit_id": event["circuit_id"],
            "scheduled_start_utc": schedule,
            "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
            "window_end_utc_exclusive": window_end.isoformat().replace("+00:00", "Z"),
            "in_window": True, "station_id": station_id,
            "station_name": station["station_name"],
            "station_latitude": station["station_latitude"],
            "station_longitude": station["station_longitude"],
            "station_elevation_m": station["station_elevation_m"],
            "distance_to_circuit_km": round(distance, 3),
            "observation_time_utc": parse_utc(str(record["DATE"])).isoformat().replace("+00:00", "Z"),
            "report_type": record.get("REPORT_TYPE", ""),
            **_measurement_columns(record.get("TMP"), "TMP"),
            **_measurement_columns(record.get("DEW"), "DEW"),
            "source_url": source_url, "consulted_at_utc": consulted_at_utc,
            "response_sha256": response_sha256,
            "quality_flags_interpreted": False,
            "duplicate_key_count_for_event_window": duplicates["duplicate_key_count"],
        })
    return rows


def build_noaa_url(station_id: str, start_date: date, end_date: date) -> str:
    params = {
        "dataset": "global-hourly", "stations": str(station_id),
        "startDate": start_date.isoformat(), "endDate": end_date.isoformat(),
        "format": "json", "includeAttributes": "true",
    }
    return f"{NOAA_ENDPOINT}?{urlencode(params)}"


def fetch_noaa_day(station_id: str, start_date: date, end_date: date) -> tuple[list[dict[str, Any]], str, str, str]:
    """Fetch a single-station, minimum-date-span NOAA Global Hourly response."""
    url = build_noaa_url(station_id, start_date, end_date)
    request = Request(url, headers={"User-Agent": "F1-Data-Lab-climate-study/3B (targeted research)"})
    consulted_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with urlopen(request, timeout=40) as response:
        body = response.read()
    body_hash = hashlib.sha256(body).hexdigest()
    decoded = json.loads(body.decode("utf-8"))
    if not isinstance(decoded, list):
        raise ValueError("NOAA response is not a JSON array")
    return decoded, url, consulted_at, body_hash


def fetch_event_windows(
    events: Sequence[Mapping[str, Any]],
    station_by_race_id: Mapping[str, Mapping[str, Any]],
    raw_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch one station/date span per event and persist only selected raw rows.

    The public endpoint accepts date bounds; response rows are locally filtered
    to the requested UTC window before any source-row snapshot is stored.
    Existing snapshot files are never overwritten.
    """
    raw_dir = Path(raw_dir)
    observation_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        station = station_by_race_id[str(event["race_id"])]
        window_start, window_end = window_for_event(str(event["scheduled_start_utc"]))
        first_date = window_start.date()
        last_date = (window_end - timedelta(microseconds=1)).date()
        queried_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        request_url = build_noaa_url(str(station["station_id"]), first_date, last_date)
        try:
            records, url, consulted_at, body_hash = fetch_noaa_day(
                str(station["station_id"]), first_date, last_date)
            selected = records_in_window(records, str(event["scheduled_start_utc"]))
            rows = observation_rows_for_event(event, station, selected, url, consulted_at, body_hash)
            snapshot_path = raw_dir / f"noaa_{event['season']}_{event['race_id']}.json"
            payload = {
                "station_id": str(station["station_id"]),
                "event_race_id": str(event["race_id"]),
                "query_url": url,
                "consulted_at_utc": consulted_at,
                "full_response_sha256": body_hash,
                "selection_window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
                "selection_window_end_utc_exclusive": window_end.isoformat().replace("+00:00", "Z"),
                "note": "Only source rows within the event window are stored; no interpolation.",
                "records": selected,
            }
            with snapshot_path.open("x", encoding="utf-8", newline="") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            observation_rows.extend(rows)
            query_rows.append({
                "season": event["season"], "grand_prix": event["grand_prix"],
                "race_id": event["race_id"], "station_id": station["station_id"],
                "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
                "window_end_utc_exclusive": window_end.isoformat().replace("+00:00", "Z"),
                "request_url": url, "consulted_at_utc": consulted_at,
                "response_sha256": body_hash, "response_records": len(records),
                "window_records": len(selected), "query_status": "ok", "error_message": "",
            })
        except Exception as exc:
            query_rows.append({
                "season": event["season"], "grand_prix": event["grand_prix"],
                "race_id": event["race_id"], "station_id": station["station_id"],
                "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
                "window_end_utc_exclusive": window_end.isoformat().replace("+00:00", "Z"),
                "request_url": request_url, "consulted_at_utc": queried_at,
                "response_sha256": "", "response_records": 0, "window_records": 0,
                "query_status": "failed", "error_message": f"{type(exc).__name__}: {exc}",
            })
        if index < len(events) - 1:
            time.sleep(0.25)
    return observation_rows, query_rows


def audit_observation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return deterministic coverage, sorting and duplicate diagnostics."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("race_id", "")), []).append(row)
    already_sorted = all(
        group == sorted(group, key=lambda row: (
            str(row.get("observation_time_utc", "")),
            str(row.get("report_type", "")),
        ))
        for group in grouped.values()
    )
    duplicate_info = audit_observation_duplicates([
        {"STATION": row.get("station_id"), "DATE": row.get("observation_time_utc"),
         "REPORT_TYPE": row.get("report_type")} for row in rows
    ])
    return {
        "record_count": len(rows),
        "sorted_within_each_event": already_sorted,
        "duplicate_key_count": duplicate_info["duplicate_key_count"],
        "duplicate_keys": duplicate_info["duplicate_keys"],
        "multiple_report_type_timestamps": duplicate_info["multiple_report_type_timestamps"],
    }


def _numeric_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(field)
        if value in (None, ""):
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def compare_seasons(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create one flat descriptive row per GP, without inferential statistics."""
    result: list[dict[str, Any]] = []
    gps = sorted({str(event["grand_prix"]) for event in events})
    for gp in gps:
        summary: dict[str, Any] = {"grand_prix": gp}
        means: dict[str, float | None] = {}
        for season in ("2013", "2014"):
            event = next((item for item in events
                          if str(item["grand_prix"]) == gp
                          and str(item["season"]) == season), None)
            group = ([row for row in observations
                      if str(row.get("race_id")) == str(event["race_id"])] if event else [])
            temp = _numeric_values(group, "air_temperature_c")
            dew = _numeric_values(group, "dew_point_c")
            times = sorted(str(row["observation_time_utc"]) for row in group)
            prefix = f"{season}_"
            summary.update({
                prefix + "race_id": event.get("race_id", "") if event else "",
                prefix + "scheduled_start_utc": event.get("scheduled_start_utc", "") if event else "",
                prefix + "scheduled_start_local": event.get("scheduled_start_local", "") if event else "",
                prefix + "utc_offset": event.get("utc_offset", "") if event else "",
                prefix + "observation_count": len(group),
                prefix + "unique_timestamp_count": len({row.get("observation_time_utc") for row in group}),
                prefix + "report_type_counts": ";".join(
                    f"{kind}:{sum(str(item.get('report_type', '')) == kind for item in group)}"
                    for kind in sorted({str(item.get("report_type", "")) for item in group})
                ),
                prefix + "temperature_count": len(temp),
                prefix + "temperature_mean_c": mean(temp) if temp else None,
                prefix + "temperature_min_c": min(temp) if temp else None,
                prefix + "temperature_max_c": max(temp) if temp else None,
                prefix + "dew_point_count": len(dew),
                prefix + "dew_point_mean_c": mean(dew) if dew else None,
                prefix + "first_observation_utc": times[0] if times else "",
                prefix + "last_observation_utc": times[-1] if times else "",
                prefix + "distance_to_circuit_km": group[0].get("distance_to_circuit_km") if group else None,
            })
            means[season] = mean(temp) if temp else None
        delta = None
        if means["2013"] is not None and means["2014"] is not None:
            delta = means["2014"] - means["2013"]
        summary["temperature_mean_difference_2014_minus_2013_c"] = delta
        summary["temperature_delta_scope"] = "station observations in fixed UTC window; descriptive only"
        result.append(summary)
    return result


def compare_common_report_types(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare only report types represented in both seasons for each GP."""
    result: list[dict[str, Any]] = []
    for gp in sorted({str(event["grand_prix"]) for event in events}):
        event_by_season = {
            str(event["season"]): event for event in events
            if str(event["grand_prix"]) == gp
        }
        if not {"2013", "2014"} <= event_by_season.keys():
            continue
        rows_by_season: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
        for season in ("2013", "2014"):
            race_id = str(event_by_season[season]["race_id"])
            by_type: dict[str, list[Mapping[str, Any]]] = {}
            for row in observations:
                if str(row.get("race_id")) == race_id:
                    by_type.setdefault(str(row.get("report_type", "")), []).append(row)
            rows_by_season[season] = by_type
        common_types = sorted(set(rows_by_season["2013"]) & set(rows_by_season["2014"]))
        for report_type in common_types:
            first = rows_by_season["2013"][report_type]
            second = rows_by_season["2014"][report_type]
            first_temp = _numeric_values(first, "air_temperature_c")
            second_temp = _numeric_values(second, "air_temperature_c")
            first_dew = _numeric_values(first, "dew_point_c")
            second_dew = _numeric_values(second, "dew_point_c")
            first_mean = mean(first_temp) if first_temp else None
            second_mean = mean(second_temp) if second_temp else None
            first_times = sorted({str(row.get("observation_time_utc", "")) for row in first})
            second_times = sorted({str(row.get("observation_time_utc", "")) for row in second})
            result.append({
                "grand_prix": gp, "report_type": report_type,
                "2013_report_count": len(first), "2014_report_count": len(second),
                "2013_temperature_count": len(first_temp),
                "2014_temperature_count": len(second_temp),
                "2013_unique_timestamp_count": len(first_times),
                "2014_unique_timestamp_count": len(second_times),
                "2013_first_observation_utc": first_times[0] if first_times else "",
                "2013_last_observation_utc": first_times[-1] if first_times else "",
                "2014_first_observation_utc": second_times[0] if second_times else "",
                "2014_last_observation_utc": second_times[-1] if second_times else "",
                "2013_temperature_mean_c": first_mean,
                "2014_temperature_mean_c": second_mean,
                "temperature_mean_difference_2014_minus_2013_c": (
                    second_mean - first_mean if first_mean is not None and second_mean is not None else None
                ),
                "2013_dew_point_mean_c": mean(first_dew) if first_dew else None,
                "2014_dew_point_mean_c": mean(second_dew) if second_dew else None,
                "comparison_scope": "common report type only; timestamps are not matched and coverage may differ; descriptive only",
            })
    return result


def _quality_bool(value: Any) -> bool:
    return value is True or str(value).strip().casefold() in {"true", "1", "yes"}


def _quality_measurement_summary(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
    value_field: str,
) -> dict[str, Any]:
    values_all = _numeric_values(rows, value_field)
    accepted = [row for row in rows if _quality_bool(row.get(f"{prefix}_qc_eligible"))]
    values_accepted = _numeric_values(accepted, value_field)
    classes = Counter(str(row.get(f"{prefix}_quality_class", "missing_class")) for row in rows)
    return {
        f"{prefix}_numeric_count_all": len(values_all),
        f"{prefix}_mean_all_c": mean(values_all) if values_all else None,
        f"{prefix}_qc_accepted_count": len(values_accepted),
        f"{prefix}_mean_qc_accepted_c": mean(values_accepted) if values_accepted else None,
        f"{prefix}_qc_excluded_numeric_count": len(values_all) - len(values_accepted),
        f"{prefix}_quality_class_counts": ";".join(
            f"{name}:{classes[name]}" for name in sorted(classes)
        ),
    }


def quality_summary_by_event(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize all numeric and QC-accepted TMP/DEW values per event."""
    rows_by_race: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations:
        rows_by_race.setdefault(str(row.get("race_id", "")), []).append(row)
    summaries: list[dict[str, Any]] = []
    for event in events:
        race_id = str(event.get("race_id", ""))
        group = rows_by_race.get(race_id, [])
        times = sorted({str(row.get("observation_time_utc", "")) for row in group if row.get("observation_time_utc")})
        report_types = Counter(str(row.get("report_type", "")) for row in group)
        summary: dict[str, Any] = {
            "season": str(event.get("season", "")),
            "grand_prix": str(event.get("grand_prix", "")),
            "race_id": race_id,
            "scheduled_start_utc": event.get("scheduled_start_utc", ""),
            "scheduled_start_local": event.get("scheduled_start_local", ""),
            "utc_offset": event.get("utc_offset", ""),
            "observation_count": len(group),
            "unique_timestamp_count": len(times),
            "first_observation_utc": times[0] if times else "",
            "last_observation_utc": times[-1] if times else "",
            "report_type_counts": ";".join(f"{kind}:{report_types[kind]}" for kind in sorted(report_types)),
            "distance_to_circuit_km": group[0].get("distance_to_circuit_km") if group else None,
        }
        summary.update(_quality_measurement_summary(group, "tmp", "air_temperature_c"))
        summary.update(_quality_measurement_summary(group, "dew", "dew_point_c"))
        summaries.append(summary)
    return summaries


def quality_summary_by_common_report_types(
    events: Sequence[Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare common report types without pairing timestamps across seasons."""
    event_by_gp_season = {
        (str(event.get("grand_prix", "")), str(event.get("season", ""))): event
        for event in events
    }
    observations_by_race_type: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in observations:
        key = (str(row.get("race_id", "")), str(row.get("report_type", "")))
        observations_by_race_type.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    gps = sorted({str(event.get("grand_prix", "")) for event in events})
    for gp in gps:
        event_2013 = event_by_gp_season.get((gp, "2013"))
        event_2014 = event_by_gp_season.get((gp, "2014"))
        if event_2013 is None or event_2014 is None:
            continue
        race_2013, race_2014 = str(event_2013.get("race_id", "")), str(event_2014.get("race_id", ""))
        types_2013 = {kind for race, kind in observations_by_race_type if race == race_2013}
        types_2014 = {kind for race, kind in observations_by_race_type if race == race_2014}
        for report_type in sorted(types_2013 & types_2014):
            group_2013 = observations_by_race_type[(race_2013, report_type)]
            group_2014 = observations_by_race_type[(race_2014, report_type)]
            times_2013 = sorted({str(row.get("observation_time_utc", "")) for row in group_2013})
            times_2014 = sorted({str(row.get("observation_time_utc", "")) for row in group_2014})
            summary: dict[str, Any] = {
                "grand_prix": gp,
                "report_type": report_type,
                "2013_report_count": len(group_2013),
                "2014_report_count": len(group_2014),
                "2013_unique_timestamp_count": len(times_2013),
                "2014_unique_timestamp_count": len(times_2014),
                "2013_first_observation_utc": times_2013[0] if times_2013 else "",
                "2013_last_observation_utc": times_2013[-1] if times_2013 else "",
                "2014_first_observation_utc": times_2014[0] if times_2014 else "",
                "2014_last_observation_utc": times_2014[-1] if times_2014 else "",
                "comparison_scope": "common type only; timestamps not matched; coverage may differ; descriptive only",
            }
            for prefix, value_field, first, second in (
                ("tmp", "air_temperature_c", group_2013, group_2014),
                ("dew", "dew_point_c", group_2013, group_2014),
            ):
                stats_2013 = _quality_measurement_summary(first, prefix, value_field)
                stats_2014 = _quality_measurement_summary(second, prefix, value_field)
                for key, value in stats_2013.items():
                    summary[f"2013_{key}"] = value
                for key, value in stats_2014.items():
                    summary[f"2014_{key}"] = value
                mean_key = f"{prefix}_mean_all_c"
                qc_mean_key = f"{prefix}_mean_qc_accepted_c"
                summary[f"{prefix}_mean_all_delta_2014_minus_2013_c"] = (
                    stats_2014[mean_key] - stats_2013[mean_key]
                    if stats_2013[mean_key] is not None and stats_2014[mean_key] is not None else None
                )
                summary[f"{prefix}_mean_qc_delta_2014_minus_2013_c"] = (
                    stats_2014[qc_mean_key] - stats_2013[qc_mean_key]
                    if stats_2013[qc_mean_key] is not None and stats_2014[qc_mean_key] is not None else None
                )
            summaries.append(summary)
    return summaries


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """Write a CSV only if the target file does not already exist."""
    path = Path(path)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_sha256_manifest(
    paths: Sequence[Path],
    root: Path,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Inventory existing files by relative path, byte count and SHA-256."""
    root = Path(root).resolve()
    entries = []
    seen: set[Path] = set()
    for path in sorted((Path(item).resolve() for item in paths), key=str):
        if path in seen:
            continue
        seen.add(path)
        relative = path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        entries.append({
            "path": relative.as_posix(),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "manifest_version": 1,
        "created_at_utc": timestamp,
        "reference_point": "current state after Phase 3B; created for Phase 3C",
        "historical_integrity_claim": False,
        "self_hash_excluded": True,
        "files": entries,
    }


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON exclusively; existing paths are never replaced."""
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
