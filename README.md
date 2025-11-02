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