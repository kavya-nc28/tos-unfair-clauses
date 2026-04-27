"""
Global configuration for paths, model names, and basic hyperparameters.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"  # for saved checkpoints
REPORTS_DIR    = PROJECT_ROOT / "reports"

# Hugging Face model names
BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"  

MAX_LENGTH = 256

ACTIVE_MODEL = "contrastive"

# Training hyperparameters
BASELINE_EPOCHS = 3          
BASELINE_LR = 2e-5           
BATCH_SIZE = 16              

CONTRASTIVE_EPOCHS = 3       
CONTRASTIVE_LR = 2e-5       
CONTRASTIVE_TEMPERATURE = 0.1  

#Evaluation thresholds (only fallbacks)
DEFAULT_THRESHOLD        = 0.30   # tuned on val set
DEFAULT_BINARY_THRESHOLD = 0.45

SEED = 42
