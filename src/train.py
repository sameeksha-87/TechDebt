import os
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score

MODEL_DIR = "models"

def train_and_select_best_model(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> Tuple[Any, Dict[str, Any]]:
    """
    Trains multiple models, evaluates ROC-AUC / F1, and saves the best model.
    """
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    }

    results = {}
    best_score = -1.0
    best_model = None
    best_model_name = ""

    print("\n--- Training Candidate Models ---")
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_preds)

        results[name] = {"ROC-AUC": round(auc, 4), "F1-Score": round(f1, 4)}
        print(f"[{name}] ROC-AUC: {auc:.4f} | F1-Score: {f1:.4f}")

        if auc > best_score:
            best_score = auc
            best_model = model
            best_model_name = name

    print(f"\n[+] Best Model Selected: {best_model_name} (ROC-AUC: {best_score:.4f})")
    
    # Save best model artifact
    joblib.dump(best_model, os.path.join(MODEL_DIR, "trained_model.pkl"))
    print(f"[+] Exported best model to {MODEL_DIR}/trained_model.pkl")

    return best_model, results