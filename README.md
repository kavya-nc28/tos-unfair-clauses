\# ToS Unfair Clause Detector



This project detects potentially unfair clauses in online Terms of Service (ToS) and assigns a \*\*severity score\*\* to highlight how intrusive each clause is compared to standard practice.



It is the final project for the \*\*Human-Centred Natural Language Processing\*\* course.



\---



\## Project goals



\- Train a baseline \*\*LegalBERT\*\* classifier on the Lippi et al. (2019) ToS dataset.

\- Extend it with \*\*contrastive learning\*\* to better separate “standard/fair” vs “intrusive/unfair” clauses.

\- Define a \*\*severity score (1–10)\*\* and simple layman labels, such as:

&#x20; - “You are good to go”

&#x20; - “Needs another look”

&#x20; - “This might be trouble”

&#x20; - “DO NOT AGREE TO THIS”

\- Build a small \*\*web UI\*\* where a user can upload a ToS (text/PDF) and see:

&#x20; - Highlighted problematic clauses

&#x20; - Clause-level severity + short explanation

&#x20; - An overall document verdict



\---




## Installation

To set up the project locally and ensure all dependencies are consistent across the team, run the following commands:

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
\## Repository structure



```text

tos-unfair-clauses/

├── README.md

├── requirements.txt

│

├── notebooks/

│   ├── 00\_colab\_tutorials/      # Seminar / example notebooks (read-only)

│   ├── 01\_data\_exploration.ipynb

│   ├── 02\_baseline\_legalbert.ipynb

│   ├── 03\_contrastive\_learning.ipynb

│   └── 04\_frontend\_integration.ipynb

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── config.py                # Paths, model names, hyperparameters

│   │

│   ├── data/

│   │   ├── \_\_init\_\_.py

│   │   ├── load\_unfair\_tos.py   # Load Lippi et al. dataset / LexGLUE UNFAIR-ToS

│   │   ├── preprocess\_tosdr.py  # (Optional) ToS;DR preprocessing

│   │   └── utils\_pdf\_text.py    # PDF → text, sentence splitting, cleaning

│   │

│   ├── models/

│   │   ├── \_\_init\_\_.py

│   │   ├── baseline\_legalbert.py      # Baseline classifier

│   │   └── contrastive\_legalbert.py   # Classifier + contrastive head

│   │

│   ├── training/

│   │   ├── \_\_init\_\_.py

│   │   ├── train\_baseline.py    # Train baseline model

│   │   ├── train\_contrastive.py # Train contrastive model

│   │   └── evaluate.py          # F1, AUC, PR-AUC, nDCG, Kendall tau, etc.

│   │

│   └── frontend/

│       ├── \_\_init\_\_.py

│       ├── severity\_mapping.py  # Model outputs → \[1–10] severity → text labels

│       └── app.py               # Gradio (or similar) UI

│

├── data/

│   ├── raw/         # Original datasets (not committed)

│   ├── interim/     # Cleaned / split CSVs

│   └── processed/   # Model-ready data

│

├── reports/

│   ├── hcnlp\_final\_report.tex   # 12-page report (ACM 1-column)

│   └── figures/

│       └── ...                  # Plots, diagrams, UI screenshots

│

├── slides/

│   └── presentation.pptx        # ≤ 30 slides

│

└── scripts/

&#x20;   ├── run\_baseline.sh          # (optional) convenience scripts

&#x20;   ├── run\_contrastive.sh

&#x20;   └── run\_app.sh


