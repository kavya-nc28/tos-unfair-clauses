# ToS Unfair Clause Detector

This project detects potentially unfair clauses in online Terms of Service (ToS) and assigns a **severity score** to highlight how intrusive each clause is compared to standard practice.

It is the final project for the **Human-Centred Natural Language Processing** course.

***

## Project Goals

- Train a baseline **LegalBERT** classifier on the UNFAIR-ToS / LexGLUE dataset.
- Extend it with **contrastive learning** to better separate "standard/fair" vs. "intrusive/unfair" clauses.
- Define a **severity score (1–10)** and simple layman labels:
  - "You are good to go"
  - "Needs another look"
  - "This might be trouble"
  - "DO NOT AGREE TO THIS"
- Build a small **web UI** where a user can upload a ToS (text/PDF) and see:
  - Highlighted problematic clauses
  - Clause-level severity + short explanation
  - An overall document verdict

***

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv

# 2. Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

***

## Running the Application

### Step 1 — Start the Backend API

```bash
python -m uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
On first start, the contrastive LegalBERT model is loaded from `models/contrastive_legal_bert.pt`.
Expected output:

```
🔍 Looking for model at: ...\models\contrastive_legal_bert.pt
✅ Model loaded. Threshold: 0.85
INFO:     Application startup complete.
```

### Step 2 — Start the Frontend UI

Open a **second terminal** (keep the backend running):

```bash
python -m src.frontend.app
```

Then open `http://127.0.0.1:7860` in your browser.
Upload a ToS as plain text or PDF and click **Analyse**.

> ⚠️ The backend must be running before the frontend is started. If the API is not up, the frontend will show a connection error.

***

## Repository Structure

```text
tos-unfair-clauses/
├── README.md
├── requirements.txt
├── api.py                             # FastAPI backend (model loading + /predict endpoint)
├── .gitattributes
│
├── notebooks/
│   └── 02_baseline_legalbert.ipynb   # Colab training notebook
│
├── data/
│   ├── raw/         # Original datasets (not committed)
│   ├── interim/     # Cleaned / split CSVs
│   └── processed/   # Model-ready data
│
├── models/
│   ├── baseline_legal_bert.pt         # Trained baseline checkpoint (not committed — too large)
│   ├── contrastive_legal_bert.pt      # Trained contrastive checkpoint (not committed — too large)
│   ├── baseline_threshold.json        # Tuned threshold for baseline model
│   └── contrastive_threshold.json     # Tuned threshold for contrastive model
│
├── reports/
│   ├── baseline_metrics.json          # Baseline test set evaluation results
│   └── contrastive_metrics.json       # Contrastive test set evaluation results
│
├── scripts/
│   ├── run_baseline.sh
│   ├── run_contrastive.sh
│   └── run_app.sh
│
├── slides/
│
└── src/
    ├── __init__.py
    ├── config.py                      # Paths, model names, hyperparameters, ACTIVE_MODEL
    │
    ├── data/
    │   ├── __init__.py
    │   ├── load_unfair_tos.py         # LexGLUE UNFAIR-ToS loader + tokenization
    │   ├── preprocess_tosdr.py        # ToS;DR preprocessing
    │   └── utils_pdf_text.py          # PDF → text, sentence splitting, cleaning
    │
    ├── models/
    │   ├── __init__.py
    │   ├── baseline_legalbert.py      # Dual-head classifier (8-label + binary)
    │   └── contrastive_legalbert.py   # Classifier + projection head (contrastive loss)
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train_baseline.py          # Train + threshold tuning + checkpoint save
    │   ├── train_contrastive.py       # Contrastive training pipeline
    │   └── evaluate.py                # F1, ROC-AUC, PR-AUC on test split
    │
    ├── inference/
    │   ├── __init__.py
    │   ├── preprocess_input.py        # Clean text + split into clauses
    │   ├── predict.py                 # Load model, run inference, return probs
    │   └── postprocess_input.py       # Probs → severity band + explanations
    │
    └── frontend/
        ├── __init__.py
        ├── severity_mapping.py        # Severity score → layman label
        └── app.py                     # Gradio UI
```

***

## Model Weights — Google Drive

The `.pt` checkpoint files (~420 MB each) are excluded from Git via `.gitignore`.
Download them from Google Drive and place them in the `models/` folder before running:

```
models/
├── baseline_legal_bert.pt
└── contrastive_legal_bert.pt
```

The threshold `.json` files are small (1 KB) and **are** committed to the repo.

***

## Switching Between Models

Set `ACTIVE_MODEL` in `src/config.py`:

```python
ACTIVE_MODEL = "contrastive"   # uses contrastive_legal_bert.pt (default)
# ACTIVE_MODEL = "baseline"    # uses baseline_legal_bert.pt
```

The API and inference pipeline read this value automatically at startup.

***

## Training

### Train Baseline

```bash
python -m src.training.train_baseline
```

### Train Contrastive Model

```bash
python -m src.training.train_contrastive
```

### Evaluate on Test Set

```bash
python -m src.training.evaluate        # baseline
python -m evaluate_contrastive         # contrastive
```

> Training was run on Google Colab (T4 GPU). CPU training is supported but slow (~10× longer).

***

## Running on Google Colab

```python
# Cell 1 — Clone repo
!git clone https://github.com/YOUR_USERNAME/tos-unfair-clauses.git
%cd tos-unfair-clauses

# Cell 2 — Install dependencies
!pip install -q -r requirements.txt

# Cell 3 — Mount Drive and copy model weights
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy('/content/drive/MyDrive/hcnlp_models/contrastive_legal_bert.pt', 'models/')
shutil.copy('/content/drive/MyDrive/hcnlp_models/baseline_legal_bert.pt',    'models/')

# Cell 4 — Train contrastive model
!python -m src.training.train_contrastive

# Cell 5 — Evaluate
!python -m evaluate_contrastive
```

***

## Results

### Baseline LegalBERT

The baseline uses a dual-head architecture: an 8-label multi-label head for unfair clause type
classification and a dedicated binary head for fair/unfair detection. Both heads are trained
jointly with `BCEWithLogitsLoss`. Separate thresholds are tuned on the validation set for each head.

| Metric | Score |
|---|---|
| Multi-label macro F1 | 0.5120 |
| Multi-label micro F1 | 0.6333 |
| Binary unfair-vs-fair F1 | 0.7397 |
| Binary ROC-AUC | **0.9674** |
| Binary PR-AUC | **0.8807** |

### Contrastive LegalBERT

Extends the baseline with a supervised contrastive projection head that explicitly separates
fair vs. unfair clause embeddings in representation space.

| Metric | Score |
|---|---|
| Binary unfair-vs-fair F1 | **0.8640** |
| Binary ROC-AUC | **0.9614** |
| Binary PR-AUC | **0.9090** |
| Best threshold | 0.85 |

The contrastive model improves binary F1 by +12.5 points over the baseline while maintaining
comparable ROC-AUC, demonstrating that contrastive training produces better-calibrated decision boundaries.

***

## Architecture Notes

- **Inference pipeline**: `preprocess_input.py` → `predict.py` → `postprocess_input.py`
- `predict.py` automatically strips `token_type_ids` when running the contrastive model (which does not use them in its `forward()` signature)
- `baseline_threshold.json` and `contrastive_threshold.json` each store a `threshold` key loaded at API startup
- The fallback mode in `api.py` returns random severity bands if the model checkpoint is missing — useful for UI development without the full model

***

## Why This Design

- UNFAIR-ToS provides supervised clause-level labels that directly match the task.
- LegalBERT is pre-trained on legal text, giving a stronger domain baseline than general BERT.
- The dual-head design separates the "what type" (8-label) and "is it unfair" (binary) signals, with separate threshold tuning for each.
- Contrastive learning encourages the encoder to learn *intrinsic unfairness features* rather than surface-level topic patterns.
- The layman severity labels (1–10 mapped to verdict strings) are designed for the non-expert user — the core principle of the Human-Centred NLP course.