# Metadata Assistant – PRL / Brama Grodzka

Asystent wspiera prace badawcze i eksperymentalne nad automatycznym wzbogacaniem metadanych dla zbiorów fotograficznych Bramy Grodzkiej – Teatru NN. Projekt ma charakter narzędziowo-badawczy i łączy analizę obrazu, wyszukiwanie wizualne oraz przygotowanie danych pomocniczych do ręcznej walidacji archiwalnej, zgodnie z zasadami GLAM (Dublin Core, OAI-PMH).

## Struktura projektu

Repozytorium ma strukturę modularną, rozdzielającą kod, dane, modele oraz wyniki analiz:

## Struktura projektu

- **src/**  
  Moduły źródłowe wykorzystywane przez notatniki i skrypty (m.in. obsługa CLIP, YOLO, OCR, słowników, spójności projektu).

- **scripts/**  
  Skrypty narzędziowe uruchamiane poza notatnikami (detekcja obiektów, wyszukiwanie obrazów, konwersje danych, operacje pomocnicze).

- **notebooks/**  
  Notatniki Jupyter realizujące kolejne etapy pipeline’u (pozyskiwanie danych, analiza CLIP, detekcja obiektów, OCR, ewaluacja, wyszukiwanie wizualne i tekstowe, przegląd wyników oraz eksport do struktur zgodnych z Dublin Core).

- **models/**  
  Modele wstępnie wytrenowane wykorzystywane w analizach (CLIP, YOLO oraz modele eksperymentalne do porównań).

- **inputs/**  
  Obrazy wejściowe wykorzystywane w testach i analizach (formaty JPG, PNG, TIFF, WEBP).

- **outputs/**  
  Wyniki przetwarzania (pliki CSV, podglądy, wycinki obiektów, wyniki wyszukiwania, dane do walidacji). Katalog nie jest wersjonowany.

- **exports/**  
  Eksporty danych w formatach przygotowanych do dalszego wykorzystania (np. CSV dla dLibra).

- **logs/**  
  Logi przebiegów pipeline’u, raporty kontrolne i zapisy audytowe dotyczące generowanych sugestii i decyzji użytkownika.

- **schemas/**  
  Schematy struktur danych i plików wyjściowych, wykorzystywane do walidacji i utrzymania spójności wyników.

- **configs/**
  Pliki konfiguracyjne określające sposób uruchamiania pipeline’u, parametry środowiskowe (lokalne / instytucjonalne), konfiguracje modeli oraz ustawienia interfejsu. Konfiguracje są oddzielone od kodu wykonawczego w celu zachowania spójności i przenośności projektu.

- **docs/**
  Materiały dokumentacyjne i architektoniczne, w tym dokumenty opisujące fazę projektową, strukturę pipeline’u, decyzje architektoniczne oraz artefakty planistyczne (np. pliki YAML). Katalog pełni rolę zaplecza dokumentacji koncepcyjnej projektu.

- **metalogic/**  
  Kod pomocniczy związany z uruchamianiem pipeline’u, logiką wykonania oraz organizacją pracy projektu.

- **secrets/**  
  Konfiguracje kluczy i dostępów (lokalne, nieprzeznaczone do wersjonowania).

- **Pliki główne**  
   `requirements.txt`, `Makefile`, `README.md`
    *(requirements.txt – definicja zależności środowiskowych projektu; Makefile – pomocnicze komendy lokalne; dokumenty architektoniczne w `docs/`)*

## Modele i metody

W projekcie wykorzystano wyłącznie modele wstępnie wytrenowane oraz rozwiązania eksperymentalne do porównań:

- **Modele główne**
	•	CLIP (OpenAI / OpenCLIP, FP16)
	•	generowanie embeddingów wizualnych i tekstowych,
	•	klasyfikacja zero-shot,
	•	wyszukiwanie obrazów na podstawie podobieństwa wizualnego i zapytań tekstowych.
	•	YOLOv8 (Ultralytics)
	•	detekcja obiektów ogólnego przeznaczenia,
	•	eksperymentalne testy modeli uczonych na ograniczonych zbiorach (np. „Syrena”, „Maluch”).

- **Modele i podejścia porównawcze / eksperymentalne**
	•	LVIS (open-vocabulary detection)
	•	OWL-ViT
	•	RT-DETR
	•	GroundingDINO + SAM

Używane wyłącznie do testów porównawczych i ewaluacyjnych, bez wdrożenia produkcyjnego.

Projekt nie obejmuje trenowania modeli na pełnych zbiorach archiwalnych ani rozpoznawania kontekstu historyczno-przestrzennego (np. identyfikacji konkretnych ulic czy wydarzeń).


## Wymagania środowiskowe

Python 3.13  
Pillow 10.4.0  
PyTorch (CPU / MPS / CUDA – zależnie od środowiska)
Ultralytics YOLO  
OpenAI CLIP / OpenCLIP

## Uruchomienie (lokalne)

source .venv/bin/activate  
pip install -r requirements.txt

## Dokumentacja

Projekt posiada dokumentację na trzech uzupełniających się poziomach:

- **Dokumentacja koncepcyjno-architektoniczna**  
  Dokument PDF opisujący założenia projektowe, decyzje architektoniczne,
  plan pipeline’u oraz kontekst instytucjonalny projektu (GLAM, Dublin Core, OAI-PMH).
  Stanowi on główne źródło opisu projektu w ujęciu badawczym i koncepcyjnym.

- **Dokumenty architektoniczne (archiwalne)**  
  Pliki YAML oraz notatki projektowe dokumentujące fazę planowania systemu, w tym plan wielośrodowiskowy, 
  ścieżkę decyzyjną oraz założenia dotyczące reprodukowalności i audytu. 
  Dokumenty te nie pełnią roli aktywnej konfiguracji wykonawczej, lecz stanowią zapis decyzji projektowych.

- **Dokumentacja techniczna repozytorium**  
  Pliki README, komentarze w kodzie, schematy struktur danych oraz konfiguracje
  runtime opisujące sposób uruchamiania i organizację pipeline’u w środowisku lokalnym.