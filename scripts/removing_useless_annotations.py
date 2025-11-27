from pathlib import Path

"""
    Czyści etykiety YOLO w data/yolo_cars/labels/{train,val} tak,
    aby zostały tylko dwie klasy:
    0 – maluch
    1 – syrena

    Wszystkie wiersze z klasami 2,3,4,5,6 są usuwane.
    Puste pliki .txt zostają jako puste (obraz bez boksów).
"""

base_dir = Path("data/yolo_cars/labels")
allowed_classes = {"0", "1"}

for split in ["train", "val"]:
    split_dir = base_dir / split
    print(f"Przetwarzam: {split_dir}")
    for txt_path in split_dir.glob("*.txt"):
        with txt_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = parts[0]
            if cls in allowed_classes:
                new_lines.append(" ".join(parts) + "\n")

        with txt_path.open("w", encoding="utf-8") as f:
            f.writelines(new_lines)