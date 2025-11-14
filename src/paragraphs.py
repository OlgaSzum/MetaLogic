from __future__ import annotations

from typing import Iterable, List, Tuple, Dict

import pandas as pd


def _bbox_to_coords(bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """
    Ujednolica współrzędne bbox.

    bbox – krotka (x1, y1, x2, y2) w pikselach

    Zwraca (x1, y1, x2, y2) jako int.
    """
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)


def group_lines_to_paragraphs(
    df_ocr: pd.DataFrame,
    max_line_gap: int = 12,
) -> pd.DataFrame:
    """
    Grupuje linie OCR w akapity w obrębie pojedynczych obrazów.

    df_ocr      – DataFrame z wynikami OCR z ocr.run_ocr_batch, wymagane kolumny:
                  'file_path', 'ocr_line_bbox', 'text'
    max_line_gap – maksymalny dopuszczalny odstęp pionowy (w pikselach)
                   między kolejnymi liniami w tym samym akapicie

    Zwraca DataFrame df_paragraphs z kolumnami:
        file_path       – ścieżka do obrazu
        paragraph_id    – numer akapitu w obrębie obrazu (1, 2, 3, …)
        text            – tekst akapitu (połączone linie z separatorami "\n")
        paragraph_bbox  – bbox akapitu (x1, y1, x2, y2) – obwiednia wszystkich linii
        n_lines         – liczba linii w akapicie
        total_chars     – łączna liczba znaków w akapicie
    """

    if df_ocr.empty:
        return pd.DataFrame(
            columns=[
                "file_path",
                "paragraph_id",
                "text",
                "paragraph_bbox",
                "n_lines",
                "total_chars",
            ]
        )

    required_cols = {"file_path", "ocr_line_bbox", "text"}
    missing = required_cols.difference(df_ocr.columns)
    if missing:
        raise ValueError(f"Brak wymaganych kolumn w df_ocr: {sorted(missing)}")

    records: List[Dict] = []

    # Przetwarzamy każdy obraz osobno
    for file_path, df_img in df_ocr.groupby("file_path"):
        if df_img.empty:
            continue

        # Ekstrahujemy współrzędne z bboxów, sortujemy po y, potem po x
        tmp = df_img.copy()
        tmp[["x1", "y1", "x2", "y2"]] = tmp["ocr_line_bbox"].apply(
            lambda b: pd.Series(_bbox_to_coords(b))
        )
        tmp = tmp.sort_values(["y1", "x1"]).reset_index(drop=True)

        paragraph_id = 0
        current_lines: List[str] = []
        current_bboxes: List[Tuple[int, int, int, int]] = []

        prev_y2: int | None = None

        for _, row in tmp.iterrows():
            text = str(row["text"]).strip()
            if not text:
                continue

            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            bbox = (x1, y1, x2, y2)

            # Nowy akapit, jeśli:
            # - to pierwsza linia, albo
            # - pionowy odstęp od poprzedniej linii jest większy niż max_line_gap
            if prev_y2 is None or (y1 - prev_y2) > max_line_gap:
                # Zapisujemy dotychczasowy akapit
                if current_lines:
                    paragraph_id += 1
                    para_text = "\n".join(current_lines)
                    xs1 = [b[0] for b in current_bboxes]
                    ys1 = [b[1] for b in current_bboxes]
                    xs2 = [b[2] for b in current_bboxes]
                    ys2 = [b[3] for b in current_bboxes]
                    para_bbox = (min(xs1), min(ys1), max(xs2), max(ys2))

                    records.append(
                        {
                            "file_path": file_path,
                            "paragraph_id": paragraph_id,
                            "text": para_text,
                            "paragraph_bbox": para_bbox,
                            "n_lines": len(current_lines),
                            "total_chars": len(para_text),
                        }
                    )

                # Rozpoczynamy nowy akapit
                current_lines = [text]
                current_bboxes = [bbox]
            else:
                # Kontynuujemy istniejący akapit
                current_lines.append(text)
                current_bboxes.append(bbox)

            prev_y2 = y2

        # Ostatni akapit dla danego obrazu
        if current_lines:
            paragraph_id += 1
            para_text = "\n".join(current_lines)
            xs1 = [b[0] for b in current_bboxes]
            ys1 = [b[1] for b in current_bboxes]
            xs2 = [b[2] for b in current_bboxes]
            ys2 = [b[3] for b in current_bboxes]
            para_bbox = (min(xs1), min(ys1), max(xs2), max(ys2))

            records.append(
                {
                    "file_path": file_path,
                    "paragraph_id": paragraph_id,
                    "text": para_text,
                    "paragraph_bbox": para_bbox,
                    "n_lines": len(current_lines),
                    "total_chars": len(para_text),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "file_path",
                "paragraph_id",
                "text",
                "paragraph_bbox",
                "n_lines",
                "total_chars",
            ]
        )

    return pd.DataFrame(records)


def summarize_paragraphs_per_image(
    df_paragraphs: pd.DataFrame,
    max_chars: int = 500,
) -> pd.DataFrame:
    """
    Agreguje akapity do poziomu jednego wiersza na obraz.

    df_paragraphs – DataFrame z kolumnami:
                    file_path, paragraph_id, text, paragraph_bbox, n_lines, total_chars
    max_chars      – maksymalna długość tekstu podglądowego na obraz

    Zwraca DataFrame z kolumnami:
        file_path
        n_paragraphs  – liczba akapitów z tekstem
        total_chars    – łączna liczba znaków we wszystkich akapitach
        sample_text    – skrócony podgląd (pierwsze akapity sklejone do max_chars)
    """

    if df_paragraphs.empty:
        return pd.DataFrame(
            columns=["file_path", "n_paragraphs", "total_chars", "sample_text"]
        )

    grp = df_paragraphs.groupby("file_path")

    def _sample_text(texts: Iterable[str]) -> str:
        """
        Buduje krótki podgląd z kilku pierwszych akapitów
        przycięty do max_chars.
        """
        parts: List[str] = []
        current_len = 0
        for t in texts:
            t = str(t).strip()
            if not t:
                continue
            if current_len >= max_chars:
                break
            remaining = max_chars - current_len
            if len(t) > remaining:
                parts.append(t[:remaining])
                current_len += remaining
                break
            else:
                parts.append(t)
                current_len += len(t)
        return "\n\n".join(parts)

    df_summary = pd.DataFrame(
        {
            "n_paragraphs": grp["paragraph_id"].count(),
            "total_chars": grp["total_chars"].sum(),
            "sample_text": grp["text"].apply(_sample_text),
        }
    )

    return df_summary.reset_index()