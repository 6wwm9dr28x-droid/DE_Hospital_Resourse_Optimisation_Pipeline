import pandas as pd
import numpy as np

from src.utils.paths import INTERIM_DIR, PROCESSED_DIR, ensure_dirs
from src.utils.helpers import standardize_columns, pick_first_present

# ---------- Kaggle cleaning ----------

def clean_kaggle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize cols, parse dates, compute WaitingDays, No_show_flag, and drop duplicates.
    """
    df = df.copy()
    df = standardize_columns(df)

    # Parse dates
    for col in ['scheduledday', 'appointmentday']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Waiting days
    if {'appointmentday', 'scheduledday'}.issubset(df.columns):
        df['waitingdays'] = (df['appointmentday'] - df['scheduledday']).dt.days
    else:
        df['waitingdays'] = np.nan

    # No_show flag (0=show, 1=no-show)
    if 'no_show' in df.columns:
        df['no_show_flag'] = df['no_show'].map({'No': 0, 'Yes': 1}).astype('float64')

    # Dedup
    subset = [c for c in ['patientid', 'appointmentid'] if c in df.columns]
    before = len(df)
    if subset:
        df = df.drop_duplicates(subset=subset)
    else:
        df = df.drop_duplicates()
    removed = before - len(df)

    for c in ['age', 'scholarship', 'hypertension', 'diabetes', 'alcoholism', 'handcap', 'sms_received']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f"  Kaggle: dropped {removed} duplicate rows.")
    return df

# ---------- HDHI cleaning ----------

def clean_hdhi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize cols, parse DOA/DOD, compute LOS, normalize outcome.
    """
    df = df.copy()
    df = standardize_columns(df)

    doa_col = pick_first_present(df, ['d_o_a', 'doa', 'date_of_admission', 'admission_date', 'd_o_a_'])
    dod_col = pick_first_present(df, ['d_o_d', 'dod', 'date_of_discharge', 'discharge_date', 'd_o_d_'])

    if doa_col:
        df[doa_col] = pd.to_datetime(df[doa_col], errors='coerce', dayfirst=True)
    if dod_col:
        df[dod_col] = pd.to_datetime(df[dod_col], errors='coerce', dayfirst=True)

    dur_col = pick_first_present(df, ['duration_of_stay', 'duration_of_stay_', 'duration'])
    # ICU duration (optional; not used yet)
    _icu_col = pick_first_present(df, ['duration_of_intensive_unit_stay', 'icu_stay', 'icu_days'])

    df['computed_los'] = np.nan
    if doa_col and dod_col:
        df['computed_los'] = (df[dod_col] - df[doa_col]).dt.days

    if dur_col in df.columns:
        df['final_los'] = pd.to_numeric(df[dur_col], errors='coerce')
        df['final_los'] = df['final_los'].fillna(df['computed_los'])
    else:
        df['final_los'] = df['computed_los']

    out_col = pick_first_present(df, ['outcome', 'disposition'])
    if out_col in df.columns:
        norm = df[out_col].astype(str).str.upper().str.strip()
        df['outcome_norm'] = norm.replace({
            'EXPIRY': 'DEATH',
            'DISCHARGE': 'DISCHARGED',
            'DAMA': 'LEFT_AGAINST_ADVICE',
            'LAMA': 'LEFT_AGAINST_ADVICE'
        })

    if 'gender' in df.columns:
        df['gender'] = df['gender'].astype(str).str.upper().str.strip()

    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    before = len(df)
    df = df.drop_duplicates()
    print(f"  HDHI: dropped {before - len(df)} duplicate rows.")

    return df

# ---------- Driver ----------

def main():
    ensure_dirs()

    k_raw = pd.read_csv(INTERIM_DIR / "kaggle_raw.csv")
    h_raw = pd.read_csv(INTERIM_DIR / "hdhi_raw.csv")

    print("🧹 Cleaning Kaggle ...")
    k = clean_kaggle(k_raw)
    print("🧹 Cleaning HDHI ...")
    h = clean_hdhi(h_raw)

    kaggle_out = PROCESSED_DIR / "kaggle_clean.csv"
    hdhi_out   = PROCESSED_DIR / "hdhi_clean.csv"
    k.to_csv(kaggle_out, index=False)
    h.to_csv(hdhi_out, index=False)

    print("\n✅ Cleaning complete")
    print(f"  Kaggle rows: {len(k):,} | columns: {len(k.columns)} | saved: {kaggle_out}")
    print(f"  HDHI   rows: {len(h):,} | columns: {len(h.columns)} | saved: {hdhi_out}")

    k_missing = k.isna().mean().round(3).sort_values(ascending=False).head(8)
    h_missing = h.isna().mean().round(3).sort_values(ascending=False).head(8)
    print("\nTop missingness (Kaggle):")
    print(k_missing.to_string())
    print("\nTop missingness (HDHI):")
    print(h_missing.to_string())

if __name__ == "__main__":
    main()
