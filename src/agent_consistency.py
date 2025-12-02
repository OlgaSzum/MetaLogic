# src/agent_consistency.py
"""
Agent spójności repozytorium:
- GIT: czystość i synchronizacja z origin/<branch>
- PATHS: audyt ścieżek w notebookach (portowalność local↔GCP)
- CSV: walidacja eksportów OCR/YOLO względem schematów w ./schemas

Uruchamianie:
    python -m src.agent_consistency --json
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import subprocess
import json
import re
import sys
from datetime import datetime

try:
    import pandas as pd
except Exception as e:
    pd = None  # CSV-checks będą WARN, jeśli brak pandas


# --- Stałe i ścieżki ---
REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
SCHEMAS_DIR = REPO_ROOT / "schemas"
DEFAULT_BRANCH = "main"

# Domyślne katalogi, które powinny istnieć w repo
CORE_DIRS = [
    "data",
    "outputs",
    "logs",
    "configs",
    "schemas",
    "models",
    "exports",
]

# Reguły ścieżek w notebookach: jeśli brak pliku konfig., użyj defaultów
DEFAULT_NOTEBOOK_RULES = {
    "allowed_prefixes": ["data/", "outputs/", "models/", "exports/"],
    "require_get_path": True,
    "gcs_allowed_in": []  # nazwy notatników, w których można używać gs:// bez warstwy IO
}

# Wzorce wykrywania niedozwolonych ścieżek
REGEX_ABS_MAC = re.compile(r'(?P<q>["\']?)(/Users/[^"\']+)(?P=q)')
REGEX_ABS_GCP = re.compile(r'(?P<q>["\']?)(/home/jupyter/[^"\']+)(?P=q)')
REGEX_WIN = re.compile(r'(?P<q>["\']?)([A-Za-z]:\\[^"\']+)(?P=q)')
REGEX_GS = re.compile(r'(gs://[^\s"\'\)]+)')
REGEX_PARENT_ESCAPE = re.compile(r'(^|\s)(\.\./)+[^ \n]+')


# --- Typ wyniku ---
@dataclass
class CheckResult:
    name: str          # np. git_clean, nb_paths, ocr_schema
    status: str        # OK | WARN | FAIL
    details: str       # krótki opis
    fix_hint: str = "" # podpowiedź naprawy


# === Pomocnicze ===
def run_git_command(args: List[str]) -> Tuple[int, str, str]:
    """Uruchamia komendę git i zwraca (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git not found"


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_logs_dir():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# === GIT ===
def check_git_clean() -> CheckResult:
    rc, out, err = run_git_command(["status", "--porcelain"])
    if rc != 0:
        return CheckResult("git_clean", "WARN", f"Nie można uruchomić git: {err or out}", "Zainstaluj git / uruchom w repo.")
    if out.strip() == "":
        return CheckResult("git_clean", "OK", "Brak niezacommitowanych zmian.")
    return CheckResult("git_clean", "FAIL", f"Niezacommitowane zmiany:\n{out}", "Zacommituj lub zstashuj zmiany.")


def check_git_sync_with_origin(branch: str = DEFAULT_BRANCH) -> List[CheckResult]:
    results: List[CheckResult] = []

    # fetch
    rc, out, err = run_git_command(["fetch", "origin"])
    if rc != 0:
        results.append(CheckResult("git_fetch", "WARN", f"fetch nieudany: {err or out}", "Sprawdź sieć/remote 'origin'."))
        return results

    rc1, local, _ = run_git_command(["rev-parse", branch])
    rc2, remote, _ = run_git_command(["rev-parse", f"origin/{branch}"])
    rc3, base, _ = run_git_command(["merge-base", branch, f"origin/{branch}"])

    if rc1 or rc2 or rc3:
        results.append(CheckResult("git_sync", "WARN", "Nie można ustalić commitów lokal/remote/base.", "Sprawdź istnienie gałęzi i remote."))
        return results

    if local == remote:
        results.append(CheckResult("git_sync", "OK", f"Lokalny == origin/{branch} ({local[:7]})."))
    elif local == base:
        results.append(CheckResult("git_sync", "FAIL", f"Lokalny za origin/{branch}.", "Wykonaj: git pull --rebase"))
    elif remote == base:
        results.append(CheckResult("git_sync", "FAIL", f"Lokalny przed origin/{branch}.", "Wykonaj: git push"))
    else:
        results.append(CheckResult("git_sync", "WARN", "Rozjazd historii (brak wspólnej bazy).", "Ręczny merge/rebase."))

    return results


# === PATHS: core katalogi i notebooki ===
def check_core_directories() -> List[CheckResult]:
    results: List[CheckResult] = []
    for d in CORE_DIRS:
        p = REPO_ROOT / d
        if p.exists() and p.is_dir():
            results.append(CheckResult("core_dir", "OK", f"Katalog istnieje: {d}"))
        else:
            results.append(CheckResult("core_dir", "WARN", f"Brak katalogu: {d}", f"Utwórz {d}/"))
    return results


def load_notebook_rules() -> Dict:
    cfg = REPO_ROOT / "configs" / "notebook_path_rules.json"
    if cfg.exists():
        try:
            return load_json(cfg)
        except Exception:
            return DEFAULT_NOTEBOOK_RULES
    return DEFAULT_NOTEBOOK_RULES


def list_notebooks() -> List[Path]:
    ignore_cfg = REPO_ROOT / "configs" / "agent_ignore.yaml"
    ignored: List[str] = []
    if ignore_cfg.exists():
        try:
            # lekka obsługa bez zależności pyyaml: wczytaj linie z prefiksem "- "
            for line in ignore_cfg.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("- "):
                    ignored.append(line[2:].strip())
        except Exception:
            pass

    nbs: List[Path] = []
    for p in REPO_ROOT.rglob("*.ipynb"):
        # ignoruj artefakty i venv
        if any(seg.startswith(".") for seg in p.parts):
            continue
        rel = p.relative_to(REPO_ROOT).as_posix()
        if any(rel.startswith(ig) for ig in ignored):
            continue
        nbs.append(p)
    return sorted(nbs)


def scan_notebook_for_paths(nb_path: Path, rules: Dict) -> List[CheckResult]:
    try:
        nb = load_json(nb_path)
    except Exception as e:
        return [CheckResult("nb_paths", "WARN", f"{nb_path.name}: nie można wczytać notatnika ({e}).")]

    cells = nb.get("cells", [])
    bad_hits: List[str] = []
    warn_hits: List[str] = []
    good_hits: int = 0

    allow_gcs = nb_path.name in rules.get("gcs_allowed_in", [])
    require_get_path = bool(rules.get("require_get_path", True))

    for ci, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source: Iterable[str] = cell.get("source", [])
        # scal na linie, ale raportuj też nr linii
        for li, line in enumerate(source, start=1):
            line_str = line.rstrip("\n")

            if "get_path(" in line_str:
                good_hits += 1

            # Shell i magic
            if line_str.strip().startswith("!"):
                if REGEX_ABS_MAC.search(line_str) or REGEX_ABS_GCP.search(line_str) or REGEX_WIN.search(line_str):
                    bad_hits.append(f'{nb_path.name} [cell {ci}, line {li}]: shell z absolutną ścieżką → {line_str.strip()}')
            if line_str.strip().startswith("%run"):
                if REGEX_ABS_MAC.search(line_str) or REGEX_ABS_GCP.search(line_str) or REGEX_WIN.search(line_str):
                    bad_hits.append(f'{nb_path.name} [cell {ci}, line {li}]: %run z absolutną ścieżką → {line_str.strip()}')

            # Niedozwolone ścieżki
            for rx, label in [
                (REGEX_ABS_MAC, "ABS_MAC"),
                (REGEX_ABS_GCP, "ABS_GCP"),
                (REGEX_WIN, "ABS_WIN"),
            ]:
                m = rx.search(line_str)
                if m:
                    bad_hits.append(f'{nb_path.name} [cell {ci}, line {li}]: {label} → {m.group(0)}')

            # gs:// bez wyjątków
            mgs = REGEX_GS.search(line_str)
            if mgs and not allow_gcs:
                warn_hits.append(f'{nb_path.name} [cell {ci}, line {li}]: gs:// poza warstwą IO → {mgs.group(0)}')

            # ../ poza root
            mpar = REGEX_PARENT_ESCAPE.search(line_str)
            if mpar:
                warn_hits.append(f'{nb_path.name} [cell {ci}, line {li}]: możliwe wyjście poza repo → {line_str.strip()}')

    results: List[CheckResult] = []
    if bad_hits:
        details = "Niedozwolone ścieżki:\n" + "\n".join(bad_hits[:20])
        if len(bad_hits) > 20:
            details += f"\n… i {len(bad_hits)-20} dalszych."
        results.append(CheckResult(
            "nb_paths", "FAIL", details,
            'Zastąp ścieżki absolutne wywołaniem get_path("…") i usuń twarde /Users…, /home/jupyter…, C:\\…'
        ))
    elif warn_hits:
        details = "Ostrzeżenia ścieżek:\n" + "\n".join(warn_hits[:20])
        if len(warn_hits) > 20:
            details += f"\n… i {len(warn_hits)-20} dalszych."
        results.append(CheckResult(
            "nb_paths", "WARN", details,
            'Przenieś gs:// do warstwy IO lub dodaj notatnik do gcs_allowed_in w configs/notebook_path_rules.json'
        ))
    else:
        status_msg = "Brak niedozwolonych wzorców."
        if require_get_path and good_hits == 0:
            results.append(CheckResult(
                "nb_paths", "WARN",
                f"{nb_path.name}: {status_msg} Nie wykryto użycia get_path().",
                "Rozważ stosowanie io_utils.get_path() dla portowalności."
            ))
        else:
            results.append(CheckResult("nb_paths", "OK", f"{nb_path.name}: {status_msg}"))

    return results


def check_notebooks_paths() -> List[CheckResult]:
    rules = load_notebook_rules()
    results: List[CheckResult] = []
    nbs = list_notebooks()
    if not nbs:
        return [CheckResult("nb_paths", "WARN", "Nie znaleziono *.ipynb w repo.")]
    for nb in nbs:
        results.extend(scan_notebook_for_paths(nb, rules))
    return results


# === CSV: schematy OCR/YOLO ===
def load_schema(schema_name: str) -> Dict | None:
    path = SCHEMAS_DIR / f"{schema_name}.schema.json"
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def check_csv_against_schema(schema: Dict, schema_tag: str) -> List[CheckResult]:
    results: List[CheckResult] = []
    if pd is None:
        return [CheckResult(f"{schema_tag}_schema", "WARN", "pandas nie jest zainstalowany.", "pip install pandas")]

    paths = schema.get("paths", [])
    required = schema.get("required_columns", [])
    numeric = schema.get("numeric_columns", [])  # opcjonalnie: kolumny, które muszą być numeryczne

    if not paths:
        return [CheckResult(f"{schema_tag}_schema", "WARN", "Brak 'paths' w schemacie.", "Uzupełnij schemas/*.schema.json")]

    for sp in paths:
        f = REPO_ROOT / sp
        if not f.exists():
            results.append(CheckResult(f"{schema_tag}_schema", "WARN", f"Brak pliku: {sp}"))
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            results.append(CheckResult(f"{schema_tag}_schema", "FAIL", f"Nie można wczytać {sp}: {e}", "Sprawdź CSV i kodowanie."))
            continue

        missing = [c for c in required if c not in df.columns]
        if missing:
            results.append(CheckResult(
                f"{schema_tag}_schema", "FAIL",
                f"{sp}: brak wymaganych kolumn: {missing}",
                "Dostosuj eksport lub schemat."
            ))
        else:
            results.append(CheckResult(f"{schema_tag}_schema", "OK", f"{sp}: kolumny wymagane obecne."))

        if numeric:
            bad_num = [c for c in numeric if c in df.columns and not is_numeric_series(df[c])]
            if bad_num:
                results.append(CheckResult(
                    f"{schema_tag}_types", "FAIL",
                    f"{sp}: kolumny nienumeryczne: {bad_num}",
                    "Konwertuj do typu numerycznego (float/int)."
                ))
            else:
                results.append(CheckResult(f"{schema_tag}_types", "OK", f"{sp}: typy numeryczne poprawne."))

    return results


def is_numeric_series(s) -> bool:
    try:
        pd.to_numeric(s.dropna(), errors="raise")
        return True
    except Exception:
        return False


def check_ocr_exports() -> List[CheckResult]:
    schema = load_schema("ocr_results")
    if schema is None:
        return [CheckResult("ocr_schema", "WARN", "Brak schemas/ocr_results.schema.json.")]
    # numeric_columns opcjonalne; jeśli potrzebne, dodaj w pliku schematu.
    return check_csv_against_schema(schema, "ocr")


def check_yolo_exports() -> List[CheckResult]:
    schema = load_schema("yolo_objects")
    if schema is None:
        return [CheckResult("yolo_schema", "WARN", "Brak schemas/yolo_objects.schema.json.")]
    # Przykładowo: confidence jako numeryczne – zdefiniuj w schemacie.
    return check_csv_against_schema(schema, "yolo")


# === Orkiestracja, raport, CLI ===
def run_all_checks(
    include_git: bool = True,
    include_nb_paths: bool = True,
    include_ocr: bool = True,
    include_yolo: bool = True
) -> List[CheckResult]:
    results: List[CheckResult] = []

    # GIT
    if include_git:
        results.append(check_git_clean())
        results.extend(check_git_sync_with_origin())

    # PATHS
    results.extend(check_core_directories())
    if include_nb_paths:
        results.extend(check_notebooks_paths())

    # CSV
    if include_ocr:
        results.extend(check_ocr_exports())
    if include_yolo:
        results.extend(check_yolo_exports())

    return results


def print_report(results: List[CheckResult]) -> None:
    # Prosta tabela tekstowa
    widths = {
        "status": 6,
        "name": 18,
    }
    print(f"{'STATUS':<{widths['status']}}  {'NAME':<{widths['name']}}  DETAILS")
    print("-" * 80)
    for r in results:
        line = f"{r.status:<{widths['status']}}  {r.name:<{widths['name']}}  {r.details}"
        print(line)
        if r.fix_hint:
            print(f"{'':<{widths['status']}}  {'':<{widths['name']}}  → {r.fix_hint}")


def save_report_json(results: List[CheckResult]) -> Path:
    ensure_logs_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = LOGS_DIR / f"agent_report_{ts}.json"
    payload = [asdict(r) for r in results]
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return outp


def exit_code_from_results(results: List[CheckResult]) -> int:
    # FAIL → 1, inaczej 0
    return 1 if any(r.status == "FAIL" for r in results) else 0


def main(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Agent spójności repozytorium.")
    p.add_argument("--no-git", action="store_true", help="Pomiń sprawdzenia GIT.")
    p.add_argument("--no-nb-paths", action="store_true", help="Pomiń audyt ścieżek w notebookach.")
    p.add_argument("--no-ocr", action="store_true", help="Pomiń walidację OCR CSV.")
    p.add_argument("--no-yolo", action="store_true", help="Pomiń walidację YOLO CSV.")
    p.add_argument("--json", action="store_true", help="Zapisz raport JSON do ./logs/.")
    args = p.parse_args(argv)

    results = run_all_checks(
        include_git=not args.no_git,
        include_nb_paths=not args.no_nb_paths,
        include_ocr=not args.no_ocr,
        include_yolo=not args.no_yolo,
    )
    print_report(results)
    if args.json:
        outp = save_report_json(results)
        print(f"\nRaport zapisany: {outp.relative_to(REPO_ROOT)}")
    return exit_code_from_results(results)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))