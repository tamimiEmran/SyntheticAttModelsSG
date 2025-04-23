#!/usr/bin/env bash
set -e

# ─── Configuration ────────────────────────────────────────
ENV_NAME="ml-env"
PYTHON_VERSION="3.9"
REQ_FILE="requirements.txt"
CHANNELS="-c defaults -c conda-forge"
# ──────────────────────────────────────────────────────────

echo "👉 Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "👉 Activating '$ENV_NAME'..."
# ensure conda command is available in non-login shells
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "👉 Installing core packages via Conda..."
conda install $CHANNELS \
  numpy>=1.21 pandas>=1.3 scipy>=1.7 scikit-learn>=1.0 \
  xgboost>=1.5 catboost>=1.0 imbalanced-learn>=0.9 \
  optuna>=2.10 matplotlib>=3.4 seaborn>=0.11 tqdm>=4.60 tables>=3.6 \
  -y

echo "👉 Upgrading pip and installing any remaining packages from $REQ_FILE..."
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo "✅ Done! To start working, run: conda activate $ENV_NAME"
