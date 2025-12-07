"""
    Renderuje podglądy z ramkami LVIS i buduje prostą galerię HTML.

    Wejście:
        - CSV z detekcjami: /Users/olga/MetaLogic/outputs/lvis_try/lvis_detections.csv
        - Obrazy:            /Users/olga/MetaLogic/inputs

    Wyjście:
        - Podglądy JPG:      /Users/olga/MetaLogic/outputs/lvis_try/preview/*.jpg
        - Galeria:           /Users/olga/MetaLogic/outputs/lvis_try/preview/index.html

    Parametry:
        - min_score – minimalny próg wizualizacji etykiety (0..1)
        - max_per_image – opcjonalny limit liczby ramek na obraz (None = bez limitu)

    Uwagi:
        - Tekst ma czarne tło pod napisem dla czytelności.
        - Rozmiar czcionki i grubość linii skalują się z rozmiarem obrazu.
"""

from pathlib import Path
import cv2
import pandas as pd
import math

# --- Ścieżki ---
INPUTS_DIR = Path("/Users/olga/MetaLogic/inputs")
CSV_PATH   = Path("/Users/olga/MetaLogic/outputs/lvis_try/lvis_detections.csv")
OUT_DIR    = Path("/Users/olga/MetaLogic/outputs/lvis_try/preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Parametry wizualizacji ---
MIN_SCORE = 0.35
MAX_PER_IMAGE = None  # np. 300 lub None

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def draw_boxes(img_bgr, det_df):
    h, w = img_bgr.shape[:2]
    # skala do czcionki i grubości linii zależna od rozmiaru obrazu
    base = (w + h) / 2.0
    thickness = max(2, int(round(0.003 * base)))
    font_scale = max(0.6, 0.0009 * base)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, r in det_df.iterrows():
        x1 = int(_clamp(r["x_min"], 0, w - 1))
        y1 = int(_clamp(r["y_min"], 0, h - 1))
        x2 = int(_clamp(r["x_max"], 0, w - 1))
        y2 = int(_clamp(r["y_max"], 0, h - 1))
        label = str(r["label"])
        score = float(r["score"])
        text = f"{label} {score:.2f}"

        # prostokąt
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (40, 200, 40), thickness)

        # tło pod napisem
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = ty1 + th + baseline + 4
        tx1 = x1
        tx2 = min(w - 1, x1 + tw + 6)
        cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)

        # napis
        cv2.putText(img_bgr, text, (x1 + 3, ty2 - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_bgr

def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["score"] >= MIN_SCORE].copy()

    # grupowanie po pliku
    groups = df.groupby("file_name", sort=True)

    saved = []
    for fname, g in groups:
        in_path = INPUTS_DIR / fname
        if not in_path.exists():
            continue

        # ewentualny limit ramek na obraz (najwyższe score)
        if isinstance(MAX_PER_IMAGE, int):
            g = g.sort_values("score", ascending=False).head(MAX_PER_IMAGE)

        img = cv2.imread(str(in_path))
        if img is None:
            continue

        img = draw_boxes(img, g)

        out_path = OUT_DIR / fname
        # gwarancja rozszerzenia .jpg (jeśli wejście było .png/.tif)
        if out_path.suffix.lower() not in {".jpg", ".jpeg"}:
            out_path = out_path.with_suffix(".jpg")

        cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved.append(out_path.name)

    # prosta galeria HTML (duże podglądy 1024px szer.)
    index_html = OUT_DIR / "index.html"
    with index_html.open("w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html lang="pl"><head><meta charset="utf-8"><title>LVIS previews</title>
<style>
body{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;margin:24px;}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px;}
.card{border:1px solid #ddd;border-radius:12px;padding:12px;}
img{max-width:100%;height:auto;display:block;border-radius:8px;}
h4{margin:0 0 8px 0;font-weight:600;font-size:14px;word-break:break-all;}
</style></head><body>
<h2>LVIS – adnotowane podglądy</h2>
<div class="grid">
""")
        for name in saved:
            f.write(f'<div class="card"><h4>{name}</h4><a href="{name}" target="_blank"><img src="{name}" loading="lazy"></a></div>\n')
        f.write("</div></body></html>")

    print(f"Zapisano {len(saved)} podglądów → {OUT_DIR}/index.html")

if __name__ == "__main__":
    main()