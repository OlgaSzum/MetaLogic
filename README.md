# Metadata Assistant – PRL / Brama Grodzka

Asystent wspiera uzupełnianie metadanych w archiwum dLibra Bramy Grodzkiej,
łącząc klasyfikację obrazów (PRL / non-PRL), wykrywanie obiektów, OCR
i sugestie opisów zgodne z zasadami GLAM (Dublin Core, OAI-PMH, AI4Culture).

## Struktura projektu

metalogic/        – kod źródłowy pipeline’u
configs/          – konfiguracje YAML
schemas/          – mapowanie pól (Dublin Core, eksport dLibra)
notebooks/        – analizy i pipeline ML
data/             – struktura katalogów na dane (uczestniczący w .gitignore)
outputs/          – wyniki (nie wersjonowane)
models/           – wagi modeli (nie wersjonowane)
logs/             – logi i audyt decyzji AI
exports/          – eksporty CSV dla dLibra

## Modele

- CLIP (FP16) – klasyfikacja obrazów i embeddingi
- YOLOv8n – detekcja obiektów
- Google Vision OCR – opcjonalnie

## Wymagania

Python 3.13  
Pillow 10.4.0  
PyTorch (MPS / CUDA)  
Ultralytics YOLO  
OpenAI CLIP

## Uruchomienie

source .venv/bin/activate  
pip install -r requirements.txt

## Dokumentacja

project_plan.yaml – główna specyfikacja projektu
docs/ – dodatkowe materiały (pipeline, schematy, logika GLAM)