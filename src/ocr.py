from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Any

import pandas as pd
from PIL import Image
from google.cloud import vision

from src.ocr_utils import generate_tiles, enhance_for_ocr


def create_vision_client(credentials_path: str | Path | None = None) -> Any:
    """
    Tworzy klienta Google Vision (ImageAnnotatorClient).

    credentials_path – ścieżka do pliku JSON z kluczem serwisowym;
                       jeśli None, używane są domyślne poświadczenia środowiska.
    """
    import os

    if credentials_path is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)

    return vision.ImageAnnotatorClient()


def load_image(path: Path) -> Image.Image:
    """
    Wczytuje obraz z dysku jako RGB.
    """
    return Image.open(path).convert("RGB")


def ocr_crop_with_boxes(
    img_crop: Image.Image,
    vision_client: Any,
    lang: Tuple[str, ...] = ("pl",),
) -> List[Dict]:
    """
    Wykonuje OCR pojedynczego kafelka obrazu i zwraca linie tekstu z bboxami.

    Zwraca listę słowników:
        {
            "text": str,
            "bbox": (x1, y1, x2, y2),
        }
    """
    import io

    buf = io.BytesIO()
    img_crop.save(buf, format="PNG")
    image = vision.Image(content=buf.getvalue())

    response = vision_client.document_text_detection(  # type: ignore[attr-defined]
        image=image,
        image_context=vision.ImageContext(language_hints=list(lang)),
    )

    out: List[Dict] = []
    if not response.full_text_annotation:
        return out

    annotation = response.full_text_annotation

    for page in annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:

                line_words: List[str] = []
                line_bbox: List[float] | None = None

                for word in para.words:
                    w_text = "".join(sym.text for sym in word.symbols)
                    if not w_text.strip():
                        continue
                    line_words.append(w_text)

                    v = word.bounding_box.vertices
                    xs = [vtx.x for vtx in v]
                    ys = [vtx.y for vtx in v]
                    wx1, wx2 = min(xs), max(xs)
                    wy1, wy2 = min(ys), max(ys)

                    if line_bbox is None:
                        line_bbox = [wx1, wy1, wx2, wy2]
                    else:
                        line_bbox[0] = min(line_bbox[0], wx1)
                        line_bbox[1] = min(line_bbox[1], wy1)
                        line_bbox[2] = max(line_bbox[2], wx2)
                        line_bbox[3] = max(line_bbox[3], wy2)

                if line_words and line_bbox is not None:
                    out.append(
                        {
                            "text": " ".join(line_words).strip(),
                            "bbox": tuple(int(v) for v in line_bbox),
                        }
                    )

    return out


def ocr_image_tiled(
    path: Path,
    vision_client: Any,
    n_cols: int = 3,
    n_rows: int = 3,
    lang: Tuple[str, ...] = ("pl",),
) -> List[Dict]:
    """
    Wykonuje OCR dla całego obrazu, dzieląc go na kafelki.

    Zwraca listę rekordów:
        {
            "file_path": str,
            "tile_bbox": (x1, y1, x2, y2),
            "ocr_line_bbox": (x1, y1, x2, y2),
            "text": str,
        }
    """
    img = load_image(path)
    tiles = generate_tiles(img, n_cols=n_cols, n_rows=n_rows)

    records: List[Dict] = []

    for tile_img, tile_bbox in tiles:
        tile_prep = enhance_for_ocr(tile_img)
        lines = ocr_crop_with_boxes(tile_prep, vision_client=vision_client, lang=lang)

        for ln in lines:
            records.append(
                {
                    "file_path": str(path),
                    "tile_bbox": tile_bbox,
                    "ocr_line_bbox": ln["bbox"],
                    "text": ln["text"],
                }
            )

    return records


def run_ocr_batch(
    image_paths: Iterable[Path],
    vision_client: Any,
    n_cols: int = 3,
    n_rows: int = 3,
    lang: Tuple[str, ...] = ("pl",),
) -> pd.DataFrame:
    """
    Uruchamia OCR tiled dla wielu obrazów i zwraca jeden DataFrame.

    Zwraca DataFrame:
        file_path, tile_bbox, ocr_line_bbox, text
    """
    all_records: List[Dict] = []
    for path in image_paths:
        recs = ocr_image_tiled(
            path=path,
            vision_client=vision_client,
            n_cols=n_cols,
            n_rows=n_rows,
            lang=lang,
        )
        all_records.extend(recs)

    if not all_records:
        return pd.DataFrame(columns=["file_path", "tile_bbox", "ocr_line_bbox", "text"])

    return pd.DataFrame(all_records)


def summarize_ocr(df_ocr: pd.DataFrame) -> pd.DataFrame:
    """
    Agreguje wyniki OCR do poziomu jednego wiersza na obraz.

    Zwraca DataFrame z kolumnami:
        file_path
        n_lines
        total_chars
        sample_text
        has_text
    """
    if df_ocr.empty:
        return pd.DataFrame(
            columns=["file_path", "n_lines", "total_chars", "sample_text", "has_text"]
        )

    grp = df_ocr.groupby("file_path")["text"]

    df_summary = pd.DataFrame(
        {
            "n_lines": grp.apply(
                lambda s: sum(bool(str(t).strip()) for t in s)
            ),
            "total_chars": grp.apply(
                lambda s: sum(len(str(t)) for t in s)
            ),
            "sample_text": grp.apply(
                lambda s: " | ".join(
                    str(t).strip() for t in s if str(t).strip()
                )[:300]
            ),
        }
    )

    df_summary["has_text"] = df_summary["n_lines"] > 0

    return df_summary.reset_index()