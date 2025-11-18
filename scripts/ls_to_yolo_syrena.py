import json
from pathlib import Path

# ---- KONFIGURACJA ---------------------------------------------------------

# Plik JSON z Label Studio
LS_JSON = Path("data/annotations/ls_merged.json")

# Folder, do którego mają powstać etykiety YOLO
OUT_LABELS = Path("data/yolo_syrena/labels/train")

# Mapowanie etykiet → ID (TO JEST KLUCZOWE)
LABEL_TO_ID = {
    "Fiat 126p": 0,
    "Syrena": 1,
}

# ---------------------------------------------------------------------------

OUT_LABELS.mkdir(parents=True, exist_ok=True)

data = json.loads(LS_JSON.read_text())

count_boxes = 0
count_images = 0

for task in data:
    image_path = task["data"]["image"]

    # Nazwa pliku obrazu
    img_name = image_path.split("/")[-1]
    stem = Path(img_name).stem

    txt_path = OUT_LABELS / f"{stem}.txt"

    boxes = []

    for ann in task["annotations"]:
        for r in ann["result"]:
            if r["type"] != "rectanglelabels":
                continue

            label = r["value"]["rectanglelabels"][0]
            class_id = LABEL_TO_ID[label]

            x = r["value"]["x"] / 100
            y = r["value"]["y"] / 100
            w = r["value"]["width"] / 100
            h = r["value"]["height"] / 100

            cx = x + w / 2
            cy = y + h / 2

            boxes.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if boxes:
        txt_path.write_text("\n".join(boxes))
        count_images += 1
        count_boxes += len(boxes)

print("=== Podsumowanie ===")
print("Plik JSON:", LS_JSON)
print("Etykiety zapisane do:", OUT_LABELS)
print("Liczba obrazów z anotacjami:", count_images)
print("Liczba boxów:", count_boxes)