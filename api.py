from __future__ import annotations
from fastapi import FastAPI
from typing import Dict
import os 
import json
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from src.inference.predict import load_model_and_tokenizer, predict_probabilities
from src.inference.postprocess_input import build_clause_results, overall_safety_score
from src.config import ACTIVE_MODEL, MODELS_DIR


app = FastAPI()

# =========================
# PATH HANDLING AND MODEL LOADING
# =========================

if ACTIVE_MODEL == "contrastive":
    CHECKPOINT      = MODELS_DIR / "contrastive_legal_bert.pt"
    THRESHOLD_FILE  = MODELS_DIR / "contrastive_threshold.json"
else:
    CHECKPOINT      = MODELS_DIR / "baseline_legal_bert.pt"
    THRESHOLD_FILE  = MODELS_DIR / "baseline_threshold.json"

THRESHOLD = 0.5
if THRESHOLD_FILE.exists():
    with open(THRESHOLD_FILE) as f:
        THRESHOLD = json.load(f).get("threshold", 0.5)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = None
tokenizer = None

# =========================
# LOAD MODEL SAFELY
# =========================
print(f"🔍 Looking for model at: {CHECKPOINT}")

try:
    if CHECKPOINT.exists():                  
        model, tokenizer = load_model_and_tokenizer(CHECKPOINT, device=device)
        print(f"✅ Model loaded. Threshold: {THRESHOLD}")
    else:
        print("⚠️ Model NOT found → fallback mode ENABLED")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model     = None
    tokenizer = None
# =========================
# API
# =========================
@app.post("/predict")
def predict(data: Dict):

    clauses = data.get("clauses", [])

    print(f"🔥 API HIT: {len(clauses)} clauses")

    if not clauses:
        return {"results": [], "safety_score": 100, "error": "No clauses provided"}

    # =========================
    # FALLBACK
    # =========================
    if model is None or tokenizer is None:
        results = [
            {
                "id":            c.get("id"),
                "text":          c.get("text"),
                "severity_band": random.choice(["CRITICAL", "HIGH", "MEDIUM", "SAFE"]),
                "severity_score": random.randint(1, 10),
                "explanation":   "Fallback mode (model not loaded)",
            }
            for c in clauses
        ]
        return {"results": results, "safety_score": 50}

    # =========================
    # REAL MODEL
    # =========================
    probs_multi, probs_binary = predict_probabilities(clauses, model, tokenizer, device=device)
    results = build_clause_results(clauses, probs_multi, threshold=THRESHOLD)
    safety  = overall_safety_score(probs_binary)

    response = {"results": results, "safety_score": safety}

    # Save result to reports/
    out_dir = Path("reports/inference_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"result_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_clauses": len(clauses),
            "safety_score": safety,
            "results": results
        }, f, indent=2)
    print(f"💾 Saved → {out_file}")

    return response