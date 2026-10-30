"""Auditoria reproduzível: python -m src.audit_data [--strict]."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform

from .data_quality import REQUIRED_FILES, audit_tables, enforce_report, DataValidationError
from .pipeline import read_csv

ROOT = Path(__file__).resolve().parents[1]


def _file_inventory(path: Path) -> dict:
    content = path.read_bytes()
    try:
        text = content.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = content.decode("ISO-8859-1")
        encoding = "ISO-8859-1"
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, [])
    rows = list(reader)
    widths = {}
    raw_missing = {column or "<unnamed>": 0 for column in header}
    for row in rows:
        widths[str(len(row))] = widths.get(str(len(row)), 0) + 1
        for index, column in enumerate(header):
            value = row[index].strip() if index < len(row) else ""
            if value in ("", "\\N", "N/A", "NA", "NULL", "null", "\x00"):
                raw_missing[column or "<unnamed>"] += 1
    inventory = {"file": path.name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                 "encoding_used": encoding, "csv_rows": len(rows), "header_fields": len(header),
                 "row_widths": widths, "raw_missing_by_column": raw_missing,
                 "nul_characters": content.count(b"\x00")}
    if "dob" in header:
        column = header.index("dob")
        formats = {"ISO_YYYY_MM_DD": 0, "DD_MM_YYYY": 0, "missing": 0, "invalid": 0}
        for row in rows:
            value = row[column].strip() if column < len(row) else ""
            if not value:
                formats["missing"] += 1
                continue
            for label, pattern in (("ISO_YYYY_MM_DD", "%Y-%m-%d"), ("DD_MM_YYYY", "%d/%m/%Y")):
                try:
                    datetime.strptime(value, pattern)
                    formats[label] += 1
                    break
                except ValueError:
                    pass
            else:
                formats["invalid"] += 1
        inventory["birth_date_formats"] = formats
    return inventory


def create_audit_report(raw_dir: Path, start_year: int = 2004, end_year: int = 2017,
                        *, tables: dict | None = None) -> dict:
    """Inclui evidência dos bytes originais; não modifica os CSVs."""
    raw_dir = Path(raw_dir)
    loaded = {} if tables is None else tables
    files, failures = {}, []
    for name in REQUIRED_FILES:
        path = raw_dir / f"{name}.csv"
        if not path.is_file():
            continue
        try:
            files[name] = _file_inventory(path)
            if tables is None:
                loaded[name] = read_csv(path)
        except (ValueError, OSError, csv.Error) as exc:
            failures.append({"code": "source_read_error", "table": name, "severity": "error",
                             "blocking": True, "count": 0, "examples": [],
                             "message": f"Não foi possível interpretar {path.name}: {type(exc).__name__}."})
    report = audit_tables(loaded, start_year, end_year)
    report["issues"].extend(failures)
    for name, inventory in files.items():
        if inventory["nul_characters"]:
            report["issues"].append({"code": "raw_nul_characters", "table": name, "severity": "warning",
                                     "blocking": False, "count": inventory["nul_characters"], "examples": [],
                                     "message": "Caracteres NUL no arquivo (count é número de caracteres); preservados no original."})
        if any(int(width) != inventory["header_fields"] for width in inventory["row_widths"]):
            report["issues"].append({"code": "csv_width_mismatch", "table": name, "severity": "warning",
                                     "blocking": False, "count": sum(count for width, count in inventory["row_widths"].items() if int(width) != inventory["header_fields"]),
                                     "examples": [], "message": "Largura de linhas difere do cabeçalho; original não reescrito."})
    report["files"] = files
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["environment"] = {"python": platform.python_version(),
                             "packages": {name: version(name) for name in ("pandas", "numpy", "scikit-learn", "streamlit", "pytest")}}
    report["provenance"] = {
        # Keep the original report fields stable: they describe the upstream
        # source and rights, which remain unconfirmed for this local copy.
        "source": None, "source_url": None, "dataset_version": None,
        "extracted_at": None, "license": None, "redistribution_permission": None,
        # Additive evidence identifies the Kaggle distribution, not its upstream
        # source or a permission to redistribute the local files.
        "identified_distribution": "Formula 1 Race Data",
        "distribution_publisher": "cjgdev",
        "distribution_url": "https://www.kaggle.com/datasets/cjgdev/formula-1-race-data-19502017",
        "upstream_source_status": "PENDENTE: fonte primária anterior ao Kaggle não identificada.",
        "declared_license": "CC BY-NC-SA 4.0 (declarada na página do dataset no Kaggle).",
        "local_file_match_evidence": (
            "Auditoria da Fase 7C: os cinco CSVs locais (results.csv, races.csv, drivers.csv, "
            "constructors.csv e status.csv) coincidiram byte a byte com os membros correspondentes "
            "do ZIP conferido; os SHA-256 também coincidiram. Evidência registrada; o dashboard "
            "não refaz essa comparação automaticamente."
        ),
        "status": (
            "Distribuição Kaggle e licença declarada identificadas; fonte primária anterior, "
            "versão/release exato e autorização de redistribuição pública permanecem pendentes."
        ),
    }
    report["validation"] = {"blocking_issues": sum(item["blocking"] for item in report["issues"]),
                            "warning_issues": sum(item["severity"] == "warning" for item in report["issues"]),
                            "policy": "Padrão bloqueia falhas estruturais/dos cálculos; --strict também reprova avisos. Nenhuma correção automática."}
    return report


def render_markdown(report: dict) -> str:
    lines = ["# Auditoria reproduzível dos dados — F1 Data Lab", "",
             f"Gerado em UTC: {report['generated_at_utc']}", "",
             "Os CSVs são somente lidos. As contagens abaixo são calculadas; não são entradas dos gráficos.", "",
             "## Contrato e políticas", "", report["validation"]["policy"], "",
             "`finish_position` continua sendo `positionOrder`. Grid positivo não comprova largada; retiradas e DSQs são preservadas.",
             "`corridas` na tabela histórica conta registros de cada grid, não necessariamente corridas distintas.",
             "Pontuação ausente permanece desconhecida; se houver qualquer ponto desconhecido no grid, `pontos_pct` é NaN, sem trocar denominador.", "",
             "## Inventário e rastreabilidade", "", "| Tabela | Registros | Anos associados a resultados | SHA-256 do CSV |", "|---|---:|---|---|"]
    for name, table in report["tables"].items():
        years = table.get("associated_result_years", {})
        lines.append(f"| {name} | {table['rows']} | {years.get('min')}–{years.get('max')} | {report['files'].get(name, {}).get('sha256', 'ausente')} |")
    coverage = report["coverage"]
    lines += ["", "As dimensões de pilotos/equipes/status não possuem ano de resultado próprio; a cobertura acima é obtida pelas referências de results.",
              "O calendário é descrito separadamente, não pelo nascimento dos pilotos.", "", "## Cobertura temporal", "", "```json", json.dumps(coverage, ensure_ascii=False, indent=2), "```", "",
              "## Seleção e exclusões explícitas", "", "```json", json.dumps(report["selection"], ensure_ascii=False, indent=2), "```", "",
              "## Ausências, duplicidades e arquivo físico", ""]
    for name, table in report["tables"].items():
        details = {"missing_by_column": table["missing_by_column"], "duplicate_rows": table["duplicate_rows"],
                   "duplicate_primary_key_rows": table.get("duplicate_primary_key_rows"), "file": report["files"].get(name)}
        lines += [f"### {name}", "", "```json", json.dumps(details, ensure_ascii=False, indent=2), "```", ""]
    checks = {"foreign_keys": report["foreign_keys"], "numeric_checks": report["numeric_checks"],
              "participations": report.get("participations", {}), "joins": report.get("joins", [])}
    lines += ["## Integridade referencial, numérica e das junções", "", "```json", json.dumps(checks, ensure_ascii=False, indent=2), "```", "",
              "## Achados", "", "As amostras são limitadas a 8 linhas por regra; as contagens consideram todas as linhas. Não são correções autorizadas.", ""]
    for item in report["issues"]:
        lines += [f"### {item['table']} — {item['code']}", "",
                  f"Severidade: **{item['severity']}**; bloqueante no modo padrão: **{item['blocking']}**; contagem: **{item['count']}**.",
                  "", item["message"], ""]
        if item["examples"]:
            lines += ["```json", json.dumps(item["examples"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Proveniência e direitos de uso", "", "```json", json.dumps(report["provenance"], ensure_ascii=False, indent=2), "```", "",
              "## Ambiente", "", "```json", json.dumps(report["environment"], ensure_ascii=False, indent=2), "```", "",
              "Reprodução: `python -m src.audit_data`. Validação estrita: `python -m src.audit_data --strict` (pode falhar pelas anomalias preservadas).", ""]
    return "\n".join(lines)


def write_audit_report(report: dict, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "data_quality.md").write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    # Evita sobrescrever CSVs originais mesmo se o usuário escolher raw como saída.
    raw = args.raw_dir.resolve()
    output = args.output_dir.resolve()
    if output == raw or raw in output.parents:
        parser.error("O diretório de saída deve ficar fora dos dados originais.")
    report = create_audit_report(raw, args.start_year, args.end_year)
    write_audit_report(report, output)
    try:
        enforce_report(report, strict=args.strict)
    except DataValidationError as exc:
        print(str(exc))
        return 1
    print(f"Auditoria gravada em {output}: {report['validation']['warning_issues']} avisos; nenhum CSV alterado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
