# TechDebt-AI 🔍

Detects technical debt and predicts which source code files are likely to experience bug fixes and maintenance bottlenecks using static code complexity and mined Git history.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Usage
Run against any local Git repository with commit history:

```bash
# 1. Unsupervised Heuristic Debt Scan (no ML required, 0-100 score)
python cli.py scan /path/to/repo --top 25

# 2. Predict Bug-Fix Probability using the Pre-Trained Multi-Repo Model
python cli.py predict /path/to/repo --top 25 --model techdebtai/model.joblib

# 3. Retrain a Custom Model on any Git Repository
python cli.py train /path/to/repo --horizon 180 --snapshots 4 --model techdebtai/model.joblib
```
*(Outputs an interactive `report.html` and raw `report.csv`.)*

---

## 🧠 Machine Learning Model & Training Dataset

* **Algorithm:** `RandomForestClassifier` (300 estimators, balanced subsampling)
* **Training Corpus:** 2,275 file-snapshots mined across 7 prominent open-source Python repositories (`requests`, `click`, `rich`, `jinja`, `httpx`, `werkzeug`, and `marshmallow`).
* **Evaluation Metrics:**
  * **ROC-AUC:** `0.844`
  * **PR-AUC:** `0.357` (vs. 10.9% base positive rate)
* **Dataset Export:** Available in [`techdebtai/multi_repo_dataset.csv`](techdebtai/multi_repo_dataset.csv).
* **Legacy Model:** The original single-repository Flask baseline is preserved in [`techdebtai/model_flask_legacy.joblib`](techdebtai/model_flask_legacy.joblib).

---

## 🔬 How It Works

1. **`git_history.py`**: Mines commit history (line churn, author distribution, commit velocity, and regex bug-fix commit heuristic).
2. **`static_metrics.py`**: Zero-checkout AST analysis via Lizard (cyclomatic complexity, function lengths, parameter counts, import coupling `fan_in` / `fan_out`, and `TODO`/`FIXME` markers).
3. **`features.py`**: Computes 19 quantitative features as of strict temporal cutoff timestamps to prevent data leakage.
4. **`model.py`**: Time-sliced cross-validation and Random Forest training/inference.
5. **`suggest.py`**: Rule-based refactoring suggestions based on top-decile outlier metrics.
6. **`report.py`**: Generates styled HTML report tables and CSV summaries.

---

## 📁 Repository Structure

```text
├── cli.py                     # Root CLI entry point
├── requirements.txt           # Python dependencies
├── .gitignore                 # Python git ignore rules
├── README.md                  # Project overview and documentation
└── techdebtai/
    ├── cli.py                 # Core CLI dispatcher
    ├── model.joblib           # Pre-trained multi-repo Random Forest model
    ├── model_flask_legacy.joblib # Legacy single-repo baseline model
    ├── multi_repo_dataset.csv # 2,275 multi-repository training dataset
    ├── report.html            # Sample generated HTML visual report
    ├── report.csv             # Sample generated CSV metrics report
    └── techdebt/
        ├── __init__.py
        ├── features.py        # 19-dimensional feature engineering
        ├── git_history.py     # Git commit and churn miner
        ├── model.py           # Snapshot dataset maker & model trainer
        ├── report.py          # HTML/CSV report generator
        ├── static_metrics.py  # Lizard AST complexity & import analyzer
        └── suggest.py         # Outlier-driven refactoring recommendations
```

---

## 📜 License
MIT License
