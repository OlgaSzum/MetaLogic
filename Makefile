# ------------------------------
#  Minimalny Makefile dla projektu
#  (lokalne środowisko, brak ciężkich sync)
# ------------------------------

# 1) Ustawienie środowiska
deps:
	@echo "Instaluję zależności..."
	pip install -r requirements.txt

# 2) Uruchomienie środowiska (przypomnienie)
env:
	@echo "Aby aktywować środowisko:"
	@echo "source .venv/bin/activate"

# 3) Czyszczenie artefaktów
clean:
	rm -rf outputs/*
	rm -rf logs/*
	rm -rf **/__pycache__/
	rm -rf **/.ipynb_checkpoints/

# 4) Synchronizacja lekka (opcjonalna) – tylko kod, bez danych
sync-light:
	@echo "Wysyłam kod do repo kopii (bez danych i modeli)..."
	rsync -av \
		--exclude 'data/' \
		--exclude 'models/' \
		--exclude 'outputs/' \
		--exclude 'logs/' \
		--exclude '__pycache__/' \
		--exclude '.ipynb_checkpoints/' \
		. /ścieżka/do/kopii

# 5) Pomoc
help:
	@echo "Dostępne komendy:"
	@echo "  make deps       – instalacja zależności"
	@echo "  make env        – przypomnienie aktywacji venv"
	@echo "  make clean      – czyszczenie artefaktów"
	@echo "  make sync-light – lekka synchronizacja kodu"