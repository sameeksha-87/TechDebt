import os
import joblib
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Extracts features/labels, splits data, scales features, and saves transformers.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Target Column Handling
    target_candidates = [col for col in df.columns if any(k in col.lower() for k in ['debt', 'label', 'target', 'bug', 'defects'])]
    if not target_candidates:
        # Fallback to last column if no target name matched
        target_col = df.columns[-1]
    else:
        target_col = target_candidates[0]

    # Convert continuous target counts into a binary label if needed (e.g. 0 = low debt/clean, >0 = debt/defects)
    y = df[target_col]
    if y.nunique() > 2:
        print(f"[!] Target column '{target_col}' has continuous values. Binarizing (> 0 -> 1)...")
        y = (y > 0).astype(int)

    # 2. Extract Feature Columns
    ignore_cols = [target_col, 'file_path', 'repo', 'file_name', 'commit_hash', 'id']
    feature_cols = [c for c in df.columns if c not in ignore_cols and df[c].dtype in ['int64', 'float64']]

    X = df[feature_cols]

    # 3. Train-Test Split (use stratify only if both classes have enough samples)
    should_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if should_stratify else None
    )

    # 4. Impute and Scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(imputer.fit_transform(X_train)),
        columns=feature_cols,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(imputer.transform(X_test)),
        columns=feature_cols,
        index=X_test.index
    )

    # 5. Save Artifacts
    joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    X_train_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_test.csv"), index=False)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols