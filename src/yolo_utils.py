from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont


def get_device() -> torch.device:
    """
    Wybiera najlepsze dostępne urządzenie PyTorch.

    Preferencja:
        1) mps  (Apple Silicon)
        2) cuda (GPU NVIDIA)
        3) cpu
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_to_yolo(device: torch.device) -> str:
    """
    Mapuje torch.device na string akceptowany przez Ultralytics YOLO.

    device – torch.device("mps" | "cuda" | "cpu")
    """
    t = str(device)
    if "mps" in t:
        return "mps"
    if "cuda" in t:
        return "cuda"
    return "cpu"


def load_yolo_model(model_path: str | Path, device: torch.device | None = None) -> YOLO:
    """
    Ładuje model YOLO (np. yolov8n.pt) na podane urządzenie.

    model_path – ścieżka do pliku .pt (np. yolov8n.pt w katalogu głównym repo)
    device     – torch.device; jeśli None, wybierany jest przez get_device()

    Zwraca obiekt modelu YOLO.
    """
    if device is None:
        device = get_device()

    model = YOLO(str(model_path))
    # Ultralytics sam zarządza urządzeniem, ale jawne .to(...) ujednolica zachowanie.
    model.to(device_to_yolo(device))
    return model


def detect_objects_on_image(
    model: YOLO,
    path: Path,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> pd.DataFrame:
    """
    Wykrywa obiekty YOLO na pojedynczym obrazie.

    model      – obiekt YOLO z load_yolo_model
    path       – ścieżka do pliku z obrazem
    conf_thres – minimalny próg confidence dla detekcji
    iou_thres  – próg IoU (NMS), zgodny z ustawieniami Ultralytics

    Zwraca DataFrame z kolumnami:
        file_path  – ścieżka do obrazu (str)
        cls_id     – indeks klasy (int)
        cls_name   – nazwa klasy (str)
        score      – confidence (float)
        x1, y1,
        x2, y2     – współrzędne bboxu w pikselach (float)
    """
    results = model.predict(
        source=str(path),
        conf=conf_thres,
        iou=iou_thres,
        verbose=False,
    )

    if not results:
        return pd.DataFrame(
            columns=["file_path", "cls_id", "cls_name", "score", "x1", "y1", "x2", "y2"]
        )

    r = results[0]
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(
            columns=["file_path", "cls_id", "cls_name", "score", "x1", "y1", "x2", "y2"]
        )

    names = r.names  # słownik {cls_id: cls_name}

    records: List[Dict] = []
    for box in boxes:
        xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))

        records.append(
            {
                "file_path": str(path),
                "cls_id": cls_id,
                "cls_name": cls_name,
                "score": conf,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
            }
        )

    return pd.DataFrame(records)


def run_yolo_batch(
    model: YOLO,
    image_paths: Iterable[Path],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> pd.DataFrame:
    """
    Uruchamia YOLO na wielu obrazach i składa wyniki w jeden DataFrame.

    model       – obiekt YOLO
    image_paths – iterowalna lista ścieżek Path
    conf_thres  – próg confidence
    iou_thres   – próg IoU

    Zwraca „długą” tabelę df_objects:
        file_path, cls_id, cls_name, score, x1, y1, x2, y2
    """
    all_records: List[pd.DataFrame] = []

    for path in image_paths:
        df_one = detect_objects_on_image(
            model=model,
            path=path,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        if not df_one.empty:
            all_records.append(df_one)

    if not all_records:
        return pd.DataFrame(
            columns=["file_path", "cls_id", "cls_name", "score", "x1", "y1", "x2", "y2"]
        )

    return pd.concat(all_records, ignore_index=True)


def summarize_objects_per_image(df_objects: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje detekcje YOLO do poziomu obrazu.

    df_objects – DataFrame z kolumnami:
                 file_path, cls_id, cls_name, score, x1, y1, x2, y2

    Zwraca DataFrame z kolumnami:
        file_path
        n_objects      – liczba wszystkich detekcji
        n_classes      – liczba różnych klas
        classes_list   – lista klas w formie stringu (np. "car; person; sign")
    """
    if df_objects.empty:
        return pd.DataFrame(
            columns=["file_path", "n_objects", "n_classes", "classes_list"]
        )

    def _classes_list(series: pd.Series) -> str:
        uniq = sorted(set(str(c) for c in series))
        return "; ".join(uniq)

    grp = df_objects.groupby("file_path")
    df_summary = pd.DataFrame(
        {
            "n_objects": grp.size(),
            "n_classes": grp["cls_name"].nunique(),
            "classes_list": grp["cls_name"].apply(_classes_list),
        }
    )

    return df_summary.reset_index()


def plot_detections_for_image(
    path: Path,
    df_objects: pd.DataFrame,
    min_score: float = 0.0,
) -> Image.Image:
    """
    Tworzy obraz z narysowanymi bboxami YOLO (do podglądu w notebooku).

    path       – ścieżka do pliku z obrazem
    df_objects – DataFrame z detekcjami dla TEGO obrazu
    min_score  – minimalny próg score do rysowania

    Zwraca obiekt PIL.Image z narysowanymi ramkami.
    """
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)

    df = df_objects[df_objects["file_path"] == str(path)]
    if min_score > 0:
        df = df[df["score"] >= min_score]

    for _, row in df.iterrows():
        x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
        cls_name = str(row["cls_name"])
        score = float(row["score"])

        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        text = f"{cls_name} {score:.2f}"
        draw.text((x1 + 2, y1 + 2), text)

    return img