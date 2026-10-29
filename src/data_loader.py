import os
import pandas as pd

HF_DATASET_URL = "https://huggingface.co/datasets/ChilliPeep/techdebt-ml-dataset/raw/main/dataset.csv"
RAW_DATA_PATH = "data/raw/dataset.csv"

def load_dataset(use_cache: bool = True) -> pd.DataFrame:
    """
    Downloads dataset from Hugging Face or loads from local cache.
    """
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    if use_cache and os.path.exists(RAW_DATA_PATH):
        print(f"[+] Loading dataset from local cache: {RAW_DATA_PATH}")
        return pd.read_csv(RAW_DATA_PATH)
    
    print(f"[+] Downloading dataset from Hugging Face...")
    df = pd.read_csv(HF_DATASET_URL)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[+] Dataset saved to {RAW_DATA_PATH}")
    return df