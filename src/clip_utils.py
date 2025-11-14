from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Any

import pandas as pd
import torch
from PIL import Image
import open_clip


def get_device() -> torch.device:
    """
    Wybiera najlepsze dostępne urządzenie dla CLIP.

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


def load_clip_model(
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: torch.device | None = None,
) -> Tuple[Any, Any, Any]:
    """
    Ładuje model CLIP, transformację obrazu i tokenizer z open_clip.

    model_name – nazwa architektury (np. "ViT-B-32")
    pretrained – wariant wag (np. "openai")
    device     – torch.device; jeśli None, wybierany jest przez get_device()

    Zwraca krotkę (model, preprocess, tokenizer).
    """
    if device is None:
        device = get_device()

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def encode_text_prompts(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Koduje listę promptów tekstowych do embeddingów CLIP.

    model     – załadowany model CLIP
    tokenizer – tokenizer z open_clip.get_tokenizer(...)
    prompts   – lista tekstów (np. nazwy scen, etykiety)
    device    – torch.device; jeśli None, używany jest model.device

    Zwraca tensora o kształcie [N, D], znormalizowanego do długości 1.
    """
    if device is None:
        device = next(model.parameters()).device

    dev = device if device is not None else torch.device("cpu")

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=dev.type != "cpu"):
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def encode_image(
    model: Any,
    preprocess: Any,
    path: Path,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Koduje pojedynczy obraz do embeddingu CLIP.

    model      – załadowany model CLIP
    preprocess – transformacja obrazu (z load_clip_model)
    path       – ścieżka do pliku z obrazem
    device     – torch.device; jeśli None, używany jest model.device

    Zwraca tensora o kształcie [D], znormalizowanego do długości 1.
    """
    if device is None:
        device = next(model.parameters()).device

    dev = device if device is not None else torch.device("cpu")

    img = Image.open(path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=dev.type != "cpu"):
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features[0]


def predict_zero_shot_scores(
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    image_path: Path,
    text_prompts: List[str],
    device: torch.device | None = None,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Oblicza zero-shot scores CLIP dla pojedynczego obrazu względem listy promptów.

    model        – model CLIP
    preprocess   – transformacja obrazu
    tokenizer    – tokenizer CLIP
    image_path   – ścieżka do obrazu
    text_prompts – lista etykiet/promptów tekstowych
    device       – urządzenie; jeśli None, używany jest model.device
    temperature  – skalowanie logitów (domyślnie 1.0)

    Zwraca DataFrame z kolumnami:
        file_path  – ścieżka do obrazu
        label      – nazwa etykiety / promptu
        score      – prawdopodobieństwo (softmax)
        logit      – surowy logit podobieństwa
    """
    if device is None:
        device = next(model.parameters()).device

    image_emb = encode_image(model, preprocess, image_path, device=device)   # [D]
    text_emb = encode_text_prompts(model, tokenizer, text_prompts, device=device)  # [N, D]

    sims = (image_emb.unsqueeze(0) @ text_emb.t()).squeeze(0)  # [N]
    if temperature != 1.0:
        sims = sims / temperature

    probs = torch.softmax(sims, dim=-1)

    data: List[Dict] = []
    for label, logit, p in zip(text_prompts, sims.tolist(), probs.tolist()):
        data.append(
            {
                "file_path": str(image_path),
                "label": str(label),
                "logit": float(logit),
                "score": float(p),
            }
        )

    df = pd.DataFrame(data)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def predict_scene_for_image(
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    image_path: Path,
    scene_labels: List[str],
    device: torch.device | None = None,
) -> Dict:
    """
    Wybiera najlepszą scenę dla pojedynczego obrazu na podstawie listy etykiet.

    Zwraca słownik:
        {
            "file_path": str,
            "scene_label": str | None,
            "scene_score": float,
        }
    """
    df_scores = predict_zero_shot_scores(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        image_path=image_path,
        text_prompts=scene_labels,
        device=device,
    )
    if df_scores.empty:
        return {
            "file_path": str(image_path),
            "scene_label": None,
            "scene_score": 0.0,
        }

    best = df_scores.iloc[0]
    return {
        "file_path": best["file_path"],
        "scene_label": best["label"],
        "scene_score": float(best["score"]),
    }


def run_clip_scene_batch(
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    image_paths: Iterable[Path],
    scene_labels: List[str],
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Uruchamia zero-shot sceny CLIP dla wielu obrazów.

    Zwraca DataFrame z kolumnami:
        file_path
        scene_label
        scene_score
    """
    records: List[Dict] = []

    for path in image_paths:
        rec = predict_scene_for_image(
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            image_path=path,
            scene_labels=scene_labels,
            device=device,
        )
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["file_path", "scene_label", "scene_score"])

    return pd.DataFrame(records)