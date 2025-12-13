"""
Normalizacja słowników: subjects (LVIS/YOLO/LVIS) i scenes (CLIP).
- Wczytuje CSV
- Standaryzuje kolumny
- Buduje stabilny klucz 'key' z EN
- Waliduje duplikaty i puste wpisy
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timezone

def _norm_key(s: str) -> str:
    """
    Normalizuje klucz EN do stabilnej postaci.
    - lower
    - trim
    - zamiana separatorów na '_'
    - usunięcie znaków spoza [a-z0-9_]
    - redukcja wielokrotnych '_'
    """
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\/\-\.\,\:\;\(\)\[\]\{\}]+", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

@dataclass(frozen=True)
class DictSpec:
    path: Path
    kind: str  # "subjects" | "scenes"

def load_dictionary(spec: DictSpec) -> pd.DataFrame:
    """
    Wczytuje słownik i normalizuje do kolumn:
    key, en, pl, active, notes, updated_at, kind
    """
    df = pd.read_csv(spec.path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]

    # heurystyki nazw kolumn (bez wymuszania Twojego obecnego formatu)
    col_en = next((c for c in df.columns if c.lower() in {"en", "subject_en", "scene_en", "label_en"}), None)
    col_pl = next((c for c in df.columns if c.lower() in {"pl", "subject_pl", "scene_pl", "label_pl"}), None)
    col_key = next((c for c in df.columns if c.lower() in {"key", "norm_key", "normalized_key"}), None)
    col_active = next((c for c in df.columns if c.lower() in {"active", "is_active", "enabled"}), None)
    col_notes = next((c for c in df.columns if c.lower() in {"notes", "note", "comment"}), None)
    col_updated = next((c for c in df.columns if c.lower() in {"updated_at", "timestamp", "ts"}), None)

    if col_en is None:
        raise ValueError(f"[{spec.kind}] Brak kolumny EN w {spec.path.name}")

    out = pd.DataFrame({
        "en": df[col_en].astype(str).str.strip(),
        "pl": df[col_pl].astype(str).str.strip() if col_pl else "",
        "notes": df[col_notes].astype(str).str.strip() if col_notes else "",
        "active": df[col_active].astype(bool) if col_active else True,
        "updated_at": df[col_updated].astype(str) if col_updated else "",
    })

    # key: preferuj istniejący, inaczej z EN
    if col_key:
        out["key"] = df[col_key].astype(str).map(_norm_key)
    else:
        out["key"] = out["en"].map(_norm_key)

    out["kind"] = spec.kind

    # sprzątanie
    out["en"] = out["en"].replace({"nan": ""})
    out["pl"] = out["pl"].replace({"nan": ""})
    out["notes"] = out["notes"].replace({"nan": ""})
    out["updated_at"] = out["updated_at"].replace({"nan": ""})

    # walidacje
    if (out["key"] == "").any():
        bad = out.loc[out["key"] == "", ["en", "pl"]].head(20)
        raise ValueError(f"[{spec.kind}] Puste 'key' po normalizacji (pierwsze 20):\n{bad}")

    dup = out["key"].duplicated(keep=False)
    if dup.any():
        bad = out.loc[dup, ["key", "en", "pl"]].sort_values("key").head(50)
        raise ValueError(f"[{spec.kind}] Duplikaty 'key' (pierwsze 50):\n{bad}")

    return out.sort_values(["kind", "key"]).reset_index(drop=True)

def save_dictionary(df: pd.DataFrame, path: Path) -> None:
    """
    Zapisuje słownik w kanonicznym układzie kolumn.
    """
    cols = ["key", "en", "pl", "active", "notes", "updated_at", "kind"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df[cols].to_csv(path, index=False)

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")