"""
LVIS Mask R-CNN inference (Detectron2) – ~1200 klas.

Wejście:  /Users/olga/MetaLogic/inputs   (JPG/PNG/TIF)
Wyjście:  /Users/olga/MetaLogic/outputs/csv/lvis_detections.csv

Kolumny CSV:
file_name, subject_en, subject_pl, score
"""

from pathlib import Path
import cv2
import pandas as pd

# ---- Ścieżki i parametry (MUSZĄ BYĆ PRZED KONFIGURACJĄ) ----
INPUTS_DIR = Path("/Users/olga/MetaLogic/inputs")
OUT_CSV    = Path("/Users/olga/MetaLogic/outputs/csv/lvis_detections.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# opcjonalny słownik tłumaczeń EN→PL
LVIS_DICT_CSV = Path("/Users/olga/MetaLogic/data/lvis_subjects_pl.csv")

CONF_TH = 0.35
DEVICE  = "cpu"  # na M2 bez dGPU – stabilniej na CPU

# ---- Konfiguracja modelu LVIS ----
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

cfg = get_cfg()
_CONFIG_CANDIDATES = [
    "LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "LVISv1-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
    "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml",
]

last_err = None
for cand in _CONFIG_CANDIDATES:
    try:
        _cfg = get_cfg()
        _cfg.merge_from_file(model_zoo.get_config_file(cand))
        weights_url = model_zoo.get_checkpoint_url(cand)  # weryfikuje dostępność wag
        _cfg.MODEL.WEIGHTS = weights_url
        _CONFIG = cand
        cfg = _cfg
        break
    except Exception as e:
        last_err = e
else:
    raise RuntimeError(f"Brak dostępnych wag dla {_CONFIG_CANDIDATES}. Ostatni błąd: {last_err}")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONF_TH
cfg.MODEL.DEVICE = DEVICE
predictor = DefaultPredictor(cfg)

# ---- Nazwy klas (LVIS) z metadanych lub fallback ---
try:
    meta_key = cfg.DATASETS.TEST[0]
    thing_classes = list(MetadataCatalog.get(meta_key).thing_classes)
except Exception:
    from detectron2.data.datasets.lvis import get_lvis_instances_meta
    if "v1" in _CONFIG.lower():
        thing_classes = get_lvis_instances_meta("lvis_v1")["thing_classes"]
    else:
        thing_classes = get_lvis_instances_meta("lvis_v0.5")["thing_classes"]

def infer_one(image_path: Path):
    im = cv2.imread(str(image_path))  # BGR
    if im is None:
        return []
    h, w = im.shape[:2]
    outputs = predictor(im)
    inst = outputs["instances"].to("cpu")

    boxes  = inst.pred_boxes.tensor.numpy() if inst.has("pred_boxes") else []
    scores = inst.scores.numpy() if inst.has("scores") else []
    cl_ids = inst.pred_classes.numpy() if inst.has("pred_classes") else []

    rows = []
    for (x1, y1, x2, y2), s, cid in zip(boxes, scores, cl_ids):
        label = thing_classes[cid] if 0 <= cid < len(thing_classes) else str(cid)
        rows.append({
            "file_name": image_path.name,
            "label": label,
            "score": float(s),
            # współrzędne już nie będą zapisywane do CSV,
            # ale zostawiamy je w records, gdyby były potrzebne diagnostycznie
            "x_min": float(x1), "y_min": float(y1),
            "x_max": float(x2), "y_max": float(y2),
            "width": int(w), "height": int(h),
        })
    return rows

def main():
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    records = []
    for p in sorted(INPUTS_DIR.iterdir()):
        if p.suffix.lower() in exts:
            records.extend(infer_one(p))

    df = pd.DataFrame(records)

    if df.empty:
        print("Brak detekcji – nie zapisano CSV.")
        return

    # 1) zostawiamy tylko file_name, label, score
    df = df[["file_name", "label", "score"]].copy()

    # 2) label → subject_en
    df = df.rename(columns={"label": "subject_en"})

    # 3) subject_pl z lokalnego słownika EN→PL, jeśli istnieje;
    #    w przeciwnym razie subject_pl = subject_en
    if LVIS_DICT_CSV.exists():
        df_dict = pd.read_csv(LVIS_DICT_CSV)
        if not {"subject_en", "subject_pl"}.issubset(df_dict.columns):
            raise ValueError(f"Plik słownika {LVIS_DICT_CSV} musi mieć kolumny 'subject_en', 'subject_pl'.")
        df = df.merge(df_dict, on="subject_en", how="left")
        df["subject_pl"] = df["subject_pl"].fillna(df["subject_en"])
    else:
        df["subject_pl"] = df["subject_en"]

    # 4) kolejność kolumn
    df = df[["file_name", "subject_en", "subject_pl", "score"]]

    df.to_csv(OUT_CSV, index=False)
    print(f"Zapisano: {OUT_CSV}  (rekordy: {len(df)})")

if __name__ == "__main__":
    main()