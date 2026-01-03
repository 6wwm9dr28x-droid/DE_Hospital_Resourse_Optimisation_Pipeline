from pathlib import Path

# Project root (repo root directory)
ROOT = Path(__file__).resolve().parents[2]

# Raw CSV paths (exact filenames you placed at repo root)
RAW_KAGGLE = ROOT / "hospital-KaggleV2-May-2016.csv"
RAW_HDHI   = ROOT / "HDHI Admission data.csv"

# Output folders
DATA_DIR      = ROOT / "data"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR   = ROOT / "figures"
TABLES_DIR    = ROOT / "tables"

def ensure_dirs():
    for p in [DATA_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
