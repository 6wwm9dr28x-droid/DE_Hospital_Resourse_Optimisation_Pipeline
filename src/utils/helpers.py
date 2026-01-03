import re
import pandas as pd

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase, replace non-alphanumeric with underscores, collapse repeats.
    Example: 'D.O.A' -> 'd_o_a'; 'TYPE OF ADMISSION-EMERGENCY/OPD' -> 'type_of_admission_emergency_opd'
    """
    new_cols = []
    for c in df.columns:
        nc = c.strip().lower()
        nc = re.sub(r'[^0-9a-z]+', '_', nc)
        nc = re.sub(r'_+', '_', nc).strip('_')
        new_cols.append(nc)
    df.columns = new_cols
    return df

def pick_first_present(df: pd.DataFrame, candidates):
    """Return the first column name from candidates that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
