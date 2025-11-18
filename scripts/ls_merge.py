#!/usr/bin/env python3
"""
Scala dwa eksporty JSON z Label Studio (lista tasków).

Wejście:
    - old_json (np. syrena.json)
    - new_json (np. maluch.json)

Wyjście:
    - merged.json

Logika:
    - klucz deduplikacji: data["image"] lub data["image_url"]
    - jeśli obraz się powtarza → zachowujemy wersję nowszą (z new_json)
    - nie zmieniamy struktury anotacji
"""

import json
import sys
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def key(task):
    """Identyfikator obrazu — LS zwykle używa 'image' albo 'image_url'."""
    return task["data"].get("image") or task["data"].get("image_url")


def main():
    if len(sys.argv) != 4:
        print("Użycie:")
        print("  python ls_merge.py old_syrena.json new_maluch.json merged.json")
        sys.exit(1)

    old_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    old = load_json(old_path)
    new = load_json(new_path)

    merged = {}

    # najpierw stary (syrena)
    for t in old:
        k = key(t)
        merged[k] = t

    # potem nowy (maluch) nadpisuje stare jeśli konflikt
    for t in new:
        k = key(t)
        merged[k] = t

    merged_list = list(merged.values())

    with open(out_path, "w") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    print("=== Podsumowanie ===")
    print(f"Stare taski: {len(old)}")
    print(f"Nowe taski:  {len(new)}")
    print(f"Po scaleniu: {len(merged_list)}")
    print(f"Zapisano:     {out_path}")


if __name__ == "__main__":
    main()