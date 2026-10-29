import os
import joblib
import pandas as pd
from typing import List

MODEL_DIR = "models"

def predict_file_risk(df_raw_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Loads saved model pipelines and scores raw feature inputs.
    """
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "trained_model.pkl"))

    X_raw = df_raw_features[feature_cols]
    X_proc = scaler.transform(imputer.transform(X_raw))

    risk_scores = model.predict_proba(X_proc)[:, 1]

    out_df = df_raw_features.copy()
    out_df['Risk_Score'] = (risk_scores * 100).round(2)
    out_df['Risk_Category'] = pd.cut(
        out_df['Risk_Score'],
        bins=[-1, 30, 70, 100],
        labels=['Low Risk', 'Moderate Debt', 'High Risk / Hotspot']
    )

    return out_df.sort_values(by='Risk_Score', ascending=False)