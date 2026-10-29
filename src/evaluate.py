import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Any

def evaluate_feature_importances(model: Any, feature_cols: List[str]) -> pd.Series:
    """
    Extracts and plots feature importances from a tree-based model.
    """
    if not hasattr(model, "feature_importances_"):
        print("[-] Model does not support feature_importances_ attribute.")
        return pd.Series()

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    print("\n--- Top 10 Technical Debt Risk Drivers ---")
    print(importances.head(10))

    # Plot
    plt.figure(figsize=(8, 5))
    importances.head(10).plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title("Key Predictors of Technical Debt")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("data/processed/feature_importance.png")
    print("[+] Plot saved to data/processed/feature_importance.png")

    return importances