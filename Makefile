# ===== MetaLogic Makefile =====
# Konfiguracja
VM := wysokozaawansowany_gmail_com@34.118.117.108
EXCL := --exclude-from='.rsyncignore'

.PHONY: help pull backup clean sync-vm sync-vm-dry clean-vm sync-gcs

help:
	@echo "Targets:"
	@echo "  pull          - git pull --ff-only"
	@echo "  backup        - kopia *.ipynb do notebooks/_backups"
	@echo "  clean         - usuń __pycache__/ i .ipynb_checkpoints/"
	@echo "  sync-vm       - rsync na VM (prawdziwy transfer)"
	@echo "  sync-vm-dry   - rsync DRY-RUN na VM (bez zmian)"
	@echo "  clean-vm      - usuń .venv i cache na VM"
	@echo "  sync-gcs      - wyślij outputs/ do GCS"

pull:
	git pull --ff-only
	git status -sb

backup:
	mkdir -p notebooks/_backups/backup_$$(date +%F_%H%M%S)
	cp -v notebooks/*.ipynb notebooks/_backups/backup_$$(date +%F_%H%M%S)/ 2>/dev/null || true

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} + ; \
	find . -name ".ipynb_checkpoints" -type d -prune -exec rm -rf {} + ; \
	rm -rf .mypy_cache .pytest_cache

sync-vm:
	rsync -av --delete $(EXCL) "$$HOME/MetaLogic/" "$(VM):~/work/MetaLogic/"

sync-vm-dry:
	rsync -av --delete $(EXCL) --dry-run "$$HOME/MetaLogic/" "$(VM):~/work/MetaLogic/"

clean-vm:
	ssh $(VM) 'rm -rf ~/work/MetaLogic/.venv ~/work/MetaLogic/__pycache__ ~/work/MetaLogic/.ipynb_checkpoints || true'

sync-gcs:
	./scripts/sync_to_gcs.sh