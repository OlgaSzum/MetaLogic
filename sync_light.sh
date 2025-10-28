#!/usr/bin/env bash
# Sync lekkich artefaktów na Maca: CSV/JSON/TSV/TXT/MD/YAML/PARQUET/IPYNB
# Źródła: (1) VM po SSH, (2) Bucket GCS
# Użycie:
#   ./sync_light.sh gc       # VM  → Mac
#   ./sync_light.sh bucket   # GCS → Mac
#   ./sync_light.sh both     # VM  → Mac, potem GCS → Mac
#   DRYRUN=1 ./sync_light.sh gc   # podgląd bez kopiowania

set -euo pipefail

IP="34.116.247.235"
USER="wysokozaawansowany_gmail_com"
SSH_KEY="$HOME/.ssh/google_compute_engine"

REMOTE_BASE="~/work/MetaLogic"
REMOTE_DIRS=("outputs" "data" "notebooks")
BUCKET="gs://nonprl-ml"

LOCAL_BASE="$HOME/MetaLogic"
MAX_SIZE="20m"  # ignoruj >20MB przy VM

INCLUDE_PATTERNS=(
  "*.csv" "*.json" "*.tsv" "*.parquet"
  "*.txt" "*.md" "*.yml" "*.yaml"
  "*.ipynb"
)

rsync_gc() {
  for d in "${REMOTE_DIRS[@]}"; do
    src="${USER}@${IP}:${REMOTE_BASE}/${d}/"
    dest="${LOCAL_BASE}/${d}"
    mkdir -p "$dest"
    args=(-av --delete --max-size="$MAX_SIZE" -e "ssh -i ${SSH_KEY}")
    args+=(--include='*/')
    for p in "${INCLUDE_PATTERNS[@]}"; do args+=(--include="$p"); done
    args+=(--exclude='*')
    if [[ "${DRYRUN:-0}" == "1" ]]; then args+=(-n); fi
    rsync "${args[@]}" "$src" "$dest/"
  done
}

gsutil_bucket() {
  EXCLUDE_REGEX='.*\.(jpg|jpeg|png|tif|tiff|webp|jp2|bmp|gif|zip|tar|gz|7z|pt|pth|onnx|npz|npy|ckpt)$'
  for d in "${REMOTE_DIRS[@]}"; do
    src="${BUCKET}/${d}"
    dest="${LOCAL_BASE}/${d}"
    mkdir -p "$dest"
    if [[ "${DRYRUN:-0}" == "1" ]]; then
      echo "[DRYRUN] gsutil -m rsync -r -x '$EXCLUDE_REGEX' $src $dest"
    else
      gsutil -m rsync -r -x "$EXCLUDE_REGEX" "$src" "$dest"
    fi
  done
}

case "${1:-}" in
  gc)     rsync_gc ;;
  bucket) gsutil_bucket ;;
  both)   rsync_gc; gsutil_bucket ;;
  *) echo "Użycie: $0 {gc|bucket|both}  (DRYRUN=1 na podgląd)"; exit 1 ;;
esac

echo "✓ Gotowe."
