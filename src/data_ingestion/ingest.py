import pandas as pd
from src.utils.paths import RAW_KAGGLE, RAW_HDHI, INTERIM_DIR, ensure_dirs

def load_kaggle() -> pd.DataFrame:
    """
    Load Kaggle appointments. We don't assume perfect types; we let cleaning handle them.
    """
    df = pd.read_csv(RAW_KAGGLE)
    return df

def load_hdhi() -> pd.DataFrame:
    """
    Load HDHI admissions. Mixed date formats will be parsed in cleaning.
    """
    df = pd.read_csv(RAW_HDHI)
    return df

def main():
    ensure_dirs()
    k = load_kaggle()
    h = load_hdhi()

    # Save raw interim snapshots (for provenance/debug)
    k.to_csv(INTERIM_DIR / "kaggle_raw.csv", index=False)
    h.to_csv(INTERIM_DIR / "hdhi_raw.csv", index=False)

    print("✅ Ingestion complete")
    print(f"  Kaggle raw rows: {len(k):,}")
    print(f"  HDHI   raw rows: {len(h):,}")
    print(f"  Saved interim to: {INTERIM_DIR}")

if __name__ == "__main__":
    main()
