from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageOps


def generate_tiles(
    img: Image.Image,
    n_cols: int = 3,
    n_rows: int = 3,
) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
    """
    Dzieli obraz na siatkę kafelków i zwraca listę (kafelek, bbox).

    img     – obraz PIL, np. wczytany przez Image.open(...)
    n_cols  – liczba kolumn kafelków
    n_rows  – liczba wierszy kafelków

    Zwraca listę krotek:
        (tile_img, (x1, y1, x2, y2))

    gdzie (x1, y1, x2, y2) to współrzędne kafelka w oryginalnym obrazie.
    """

    if n_cols <= 0 or n_rows <= 0:
        raise ValueError("n_cols i n_rows muszą być większe od zera.")

    width, height = img.size
    tile_w = width // n_cols
    tile_h = height // n_rows

    tiles: List[Tuple[Image.Image, Tuple[int, int, int, int]]] = []

    for row in range(n_rows):
        for col in range(n_cols):
            x1 = col * tile_w
            y1 = row * tile_h

            # Ostatnia kolumna/wiersz idzie do końca obrazu,
            # żeby uniknąć utraty pikseli przy dzieleniu całkowitym.
            if col == n_cols - 1:
                x2 = width
            else:
                x2 = x1 + tile_w

            if row == n_rows - 1:
                y2 = height
            else:
                y2 = y1 + tile_h

            bbox = (x1, y1, x2, y2)
            tile_img = img.crop(bbox)
            tiles.append((tile_img, bbox))

    return tiles


def enhance_for_ocr(img: Image.Image) -> Image.Image:
    """
    Wzmacnia obraz na potrzeby OCR (na kopii, bez modyfikacji oryginału).

    img – obraz PIL (np. pojedynczy kafelek z generate_tiles)

    Operacje:
        - konwersja do skali szarości
        - autokontrast, aby uwydatnić tekst

    Zwraca nowy obiekt Image.Image, gotowy do wysłania do silnika OCR.
    """

    # pracujemy na kopii, żeby nie nadpisać oryginału
    gray = img.convert("L")

    # lekkie wzmocnienie kontrastu (bez „wybielania” całości obrazu)
    enhanced = ImageOps.autocontrast(gray)

    return enhanced