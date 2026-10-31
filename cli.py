"""Root entry point for TechDebt-AI.

Usage:
  python cli.py scan    /path/to/repo [--top 25]
  python cli.py predict /path/to/repo [--top 25] [--model techdebtai/model.joblib]
  python cli.py train   /path/to/repo [--horizon 180] [--snapshots 4]
"""
import sys
from pathlib import Path

# Add techdebtai directory to search path
techdebtai_dir = Path(__file__).resolve().parent / "techdebtai"
sys.path.insert(0, str(techdebtai_dir))

import cli

if __name__ == "__main__":
    cli.main()
