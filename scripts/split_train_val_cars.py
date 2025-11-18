from pathlib import Path
import random
import shutil

"""
Dzieli YOLO-dataset na train/val (90/10) zachowując pary obraz–etykieta.

Zakłada strukturę:
    ~/MetaLogic/data/yolo_cars/
        images/train/
        images/val/
        labels/train/
        labels/val/

Obrazy bez pliku .txt traktowane są jako negatives (tło).
"""

VAL_FRACTION = 0.10

data_root = Path.home() / "MetaLogic" / "data" / "yolo_cars"
img_train = data_root / "images" / "train"
lbl_train = data_root / "labels" / "train"
img_val = data_root / "images" / "val"
lbl_val = data_root / "labels" / "val"

img_val.mkdir(parents=True, exist_ok=True)
lbl_val.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

images = [p for p in img_train.iterdir() if p.suffix.lower() in exts]
images.sort()
random.seed(42)
random.shuffle(images)

n_total = len(images)
n_val = max(1, int(round(n_total * VAL_FRACTION)))

val_images = images[:n_val]

print(f"Liczba obrazów łącznie: {n_total}")
print(f"Przenoszę do val: {n_val}")

for img in val_images:
    # przenieś obraz
    shutil.move(str(img), img_val / img.name)

    # przenieś etykietę jeśli istnieje
    lbl = lbl_train / (img.stem + ".txt")
    if lbl.exists():
        shutil.move(str(lbl), lbl_val / lbl.name)

print("Gotowe.")
print(f"Pozostało w train: {len(list(img_train.iterdir()))} obrazów")
print(f"W val: {len(list(img_val.iterdir()))} obrazów")