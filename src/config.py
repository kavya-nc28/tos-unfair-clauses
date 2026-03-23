"""
Global configuration for paths, model names, and basic hyperparameters.
"""

from pathlib import Path

# TODO: adjust these as needed
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"  # for saved checkpoints

# Hugging Face model names
BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  # TODO: confirm choice

# Training hyperparameters
BASELINE_EPOCHS = 3          # TODO: tune
BASELINE_LR = 2e-5           # TODO: tune
BATCH_SIZE = 16              # TODO: tune

CONTRASTIVE_EPOCHS = 3       # TODO: tune
CONTRASTIVE_LR = 2e-5        # TODO: tune
CONTRASTIVE_TEMPERATURE = 0.1  # TODO: tune

SEED = 42
