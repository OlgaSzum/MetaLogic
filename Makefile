.PHONY: local cloud sync-vm sync-gcs deps

local:
	ML_CONFIG=configs/runtime.local.yaml code .

cloud:
	ssh -i ~/.ssh/google_compute_engine \
		wysokozaawansowany_gmail_com@34.118.117.108 \
		'cd ~/work/MetaLogic && git pull && source .venv/bin/activate && jupyter lab --no-browser'

sync-vm:
	./scripts/sync_to_vm.sh

sync-gcs:
	ssh -i ~/.ssh/google_compute_engine \
		wysokozaawansowany_gmail_com@34.118.117.108 \
		'cd ~/work/MetaLogic && ./scripts/sync_to_gcs.sh'

deps:
	pip install -r requirements.txt

backup:
	mkdir -p notebooks/_backup
	cp -v notebooks/*.ipynb notebooks/_backup/ || true
	echo "✅ Notebooki zapisane w notebooks/_backup"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	rm -rf outputs/* || true
	echo "🧹 Cache i outputs wyczyszczone"

pull:
	git pull --ff-only
	echo "✅ Repozytorium zaktualizowane z GitHuba"
	git status -sb
