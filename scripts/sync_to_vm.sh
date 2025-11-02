#!/usr/bin/env bash
set -euo pipefail
VM=wysokozaawansowany_gmail_com@34.118.117.108

rsync -av --delete \
  --exclude '.git' \
  --exclude '.gitignore' \
  --exclude '.venv' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '.vscode/' \
  --exclude '*.pyc' \
  --exclude 'outputs/' \
  --exclude 'notebooks/weights/' \
  "$HOME/MetaLogic/" "$VM:~/work/MetaLogic/"