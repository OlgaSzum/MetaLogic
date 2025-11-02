# MetaLogic
Asystent metadanych: AI + reguły dla archiwów (PRL / non-PRL / inne).

## Automatyzacja (Makefile)

| Komenda         | Działanie                                         |
| --------------- | ------------------------------------------------- |
| `make local`    | Uruchamia projekt lokalnie w VS Code              |
| `make sync-vm`  | Synchronizuje pliki z Maca do instancji GC        |
| `make cloud`    | Uruchamia Jupyter na instancji                    |
| `make backup`   | Kopiuje lokalne notebooki do `notebooks/_backup/` |
| `make clean`    | Czyści cache i katalog `outputs`                  |
| `make sync-gcs` | Wysyła wyniki z instancji do Cloud Storage        |
| `make deps`     | Instaluje zależności z `requirements.txt`         |

## Kolejność pracy z notatnikami

1. **01_vision_paddle_pipeline.ipynb**  
   OCR, grupowanie tekstów, kafelkowanie i analiza wizualna zdjęć.  
   Zapisuje wyniki (`_ocr.json`, `_full.json`) w katalogu `outputs/ocr/`.

2. **02_objects.ipynb**  
   Wykrywanie obiektów i logotypów (Google Vision OBJECT_LOCALIZATION + LOGO_DETECTION).  
   Wykorzystuje obrazy z katalogu `inputs/` oraz zapisuje wizualizacje i dane do `outputs/ocr/`.

💡 *Uruchamiaj notatniki w tej kolejności — drugi notebook korzysta z danych przygotowanych przez pierwszy.*