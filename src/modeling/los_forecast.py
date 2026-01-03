import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Dict, List

from src.utils.paths import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs

def load_hdhi_clean() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "hdhi_clean.csv")
    if "final_los" not in df.columns:
        if "computed_los" in df.columns:
            df["final_los"] = pd.to_numeric(df["computed_los"], errors="coerce")
        else:
            df["final_los"] = np.nan
    else:
        df["final_los"] = pd.to_numeric(df["final_los"], errors="coerce")

    if "type_of_admission_emergency_opd" not in df.columns:
        df["type_of_admission_emergency_opd"] = np.nan
    if "outcome_norm" not in df.columns and "outcome" in df.columns:
        df["outcome_norm"] = df["outcome"].astype(str).str.upper().str.strip().replace({
            "EXPIRY":"DEATH","DISCHARGE":"DISCHARGED","DAMA":"LEFT_AGAINST_ADVICE","LAMA":"LEFT_AGAINST_ADVICE"
        })
    elif "outcome_norm" not in df.columns:
        df["outcome_norm"] = np.nan

    for c in ["d_o_a","d_o_d"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    df["d_o_a_dt"] = pd.to_datetime(df.get("d_o_a"), errors="coerce").dt.normalize()
    df["d_o_d_dt"] = pd.to_datetime(df.get("d_o_d"), errors="coerce").dt.normalize()

    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    else:
        df["age"] = np.nan
    return df

def make_los_distribution_figure(h: pd.DataFrame) -> Path:
    df = h.copy()
    los = pd.to_numeric(df["final_los"], errors="coerce")
    df = df.assign(final_los=los).loc[los.notna() & (los >= 0)]
    adm = df.get("type_of_admission_emergency_opd", pd.Series(np.nan, index=df.index)).fillna("Unknown")
    out = df.get("outcome_norm", pd.Series(np.nan, index=df.index)).fillna("Unknown")

    cap = np.nanpercentile(df["final_los"], 95) if len(df["final_los"]) else 0
    df_plot = df.assign(final_los=np.minimum(df["final_los"], cap))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    groups_a, labels_a = [], []
    for k, g in df_plot.groupby(adm):
        vals = g["final_los"].dropna().values
        if len(vals) > 0:
            groups_a.append(vals); labels_a.append(str(k))
    if groups_a:
        axes[0].violinplot(groups_a, showmeans=True, showmedians=True)
        axes[0].set_xticks(range(1, len(labels_a)+1), labels_a, rotation=20)
        axes[0].set_title("LOS by Admission Type"); axes[0].set_ylabel("Length of Stay (days)")

    groups_b, labels_b = [], []
    for k, g in df_plot.groupby(out):
        vals = g["final_los"].dropna().values
        if len(vals) > 0:
            groups_b.append(vals); labels_b.append(str(k))
    if groups_b:
        axes[1].violinplot(groups_b, showmeans=True, showmedians=True)
        axes[1].set_xticks(range(1, len(labels_b)+1), labels_b, rotation=20)
        axes[1].set_title("LOS by Outcome")

    fig.suptitle("Length of Stay (LOS) distributions", fontsize=12)
    out_path = FIGURES_DIR / "RQ5_Fig1.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight"); plt.close(fig)
    return out_path

def build_occupancy_series(h: pd.DataFrame) -> pd.Series:
    """
    Event-based occupancy with correct day accounting:
    +1 on admission day, -1 on the day AFTER discharge (discharge day is the last occupied day).
    Clip at 0 to avoid negatives caused by truncated windows.
    """
    add = h["d_o_a_dt"].dropna()
    sub = (h["d_o_d_dt"].dropna() + pd.Timedelta(days=1))  # day after discharge

    if add.empty and sub.empty:
        return pd.Series(dtype="int64")

    ev = []
    if len(add): ev.append(pd.DataFrame({"Date": add, "delta": 1}))
    if len(sub): ev.append(pd.DataFrame({"Date": sub, "delta": -1}))
    ev = pd.concat(ev).groupby("Date", as_index=False)["delta"].sum().sort_values("Date")

    full_range = pd.date_range(ev["Date"].min(), ev["Date"].max(), freq="D")
    df_occ = pd.DataFrame({"Date": full_range})
    df_occ = df_occ.merge(ev, on="Date", how="left").fillna({"delta": 0})
    # clip at 0 for safety
    df_occ["OccupiedBeds"] = df_occ["delta"].cumsum().clip(lower=0).astype(int)
    return pd.Series(df_occ["OccupiedBeds"].values, index=df_occ["Date"], name="OccupiedBeds")

def ma_forecast(series: pd.Series, horizon: int = 14, window: int = 7) -> Tuple[pd.Series, pd.Series]:
    if series.empty:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64")
    ma = series.rolling(window=window, min_periods=max(1, window//2)).mean()
    last_ma = ma.dropna().iloc[-1] if not ma.dropna().empty else float(series.iloc[-window:].mean())
    # enforce non-negative forecast
    last_ma = max(0.0, float(last_ma))
    future_idx = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    fcst = pd.Series([last_ma]*horizon, index=future_idx, name="Forecast").clip(lower=0)
    return ma, fcst

def backtest_ma(series: pd.Series, horizon: int = 14, window: int = 7, folds: int = 4, step: int = 14) -> pd.DataFrame:
    rows = []
    if len(series) < (window + horizon + folds*step):
        folds = max(1, min(folds, int((len(series) - (window + horizon)) / max(1, step))))
    last = series.index.max()
    for i in range(folds, 0, -1):
        T = last - pd.Timedelta(days=i*step)
        hist = series[series.index <= T]
        future = series[(series.index > T) & (series.index <= T + pd.Timedelta(days=horizon))]
        if len(hist) < window or len(future) == 0:
            continue
        ma, fc = ma_forecast(hist, horizon=len(future), window=window)
        fc = fc.iloc[:len(future)]
        y, yhat = future.values.astype(float), fc.values.astype(float)
        mae = float(np.mean(np.abs(yhat - y)))
        rmse = float(np.sqrt(np.mean((yhat - y)**2)))
        denom = (np.abs(yhat) + np.abs(y))
        smape = float(np.mean(2.0 * np.abs(yhat - y) / np.where(denom == 0, 1.0, denom)))
        rows.append({"fold_origin": T.date().isoformat(),
                     "horizon_days": len(future), "MAE": round(mae,3),
                     "RMSE": round(rmse,3), "sMAPE": round(smape*100,2)})
    return pd.DataFrame(rows)

def make_occupancy_figure(series: pd.Series, ma: pd.Series, fc: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    if not series.empty:
        ax.plot(series.index, series.values, label="Occupied beds (daily)", alpha=0.7)
    if not ma.empty:
        ax.plot(ma.index, ma.values, label="7-day moving average", color="orange", lw=2)
    if not fc.empty:
        ax.plot(fc.index, fc.values, "--", label="14-day MA forecast", color="red")
    ax.set_title("Hospital Occupancy — Actuals & 14-day MA forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Occupied beds"); ax.legend()
    out_path = FIGURES_DIR / "RQ5_Fig2.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight"); plt.close(fig)
    return out_path

def los_summary_tables(h: pd.DataFrame) -> Path:
    df = h.copy()
    df["final_los"] = pd.to_numeric(df["final_los"], errors="coerce")
    df = df[df["final_los"].notna() & (df["final_los"] >= 0)]

    # Age bands
    if "age" in df.columns:
        bins   = [0,12,18,30,45,60,75,90,120]
        labels = ["0-11","12-17","18-29","30-44","45-59","60-74","75-89","90+"]
        df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    else:
        df["age_band"] = "Unknown"

    def agg_stats(group: pd.Series) -> Dict[str, float]:
        if group.empty: return {"count":0,"mean":np.nan,"median":np.nan,"q1":np.nan,"q3":np.nan,"iqr":np.nan}
        q1, q3 = np.nanpercentile(group, [25, 75])
        return {"count": int(group.count()),
                "mean": float(np.nanmean(group)),
                "median": float(np.nanmedian(group)),
                "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}

    # Create explicit columns for grouping (avoids 'level_1')
    df["admission_type"] = df.get("type_of_admission_emergency_opd").fillna("Unknown")
    df["outcome_clean"]  = df.get("outcome_norm").fillna("Unknown")

    by_age = df.groupby("age_band")["final_los"].apply(agg_stats).apply(pd.Series).reset_index()
    by_adm = df.groupby("admission_type")["final_los"].apply(agg_stats).apply(pd.Series).reset_index()
    by_out = df.groupby("outcome_clean")["final_los"].apply(agg_stats).apply(pd.Series).reset_index().rename(columns={"outcome_clean":"outcome"})

    out_path = TABLES_DIR / "RQ5_Table1.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        by_age.to_excel(w, index=False, sheet_name="LOS_by_AgeBand")
        by_adm.to_excel(w, index=False, sheet_name="LOS_by_AdmissionType")
        by_out.to_excel(w, index=False, sheet_name="LOS_by_Outcome")
    return out_path

def forecast_and_accuracy(series: pd.Series, horizon: int = 14, window: int = 7, folds: int = 4, step: int = 14) -> Tuple[Path, pd.Series, pd.Series]:
    ma, fc = ma_forecast(series, horizon=horizon, window=window)
    bt = backtest_ma(series, horizon=horizon, window=window, folds=folds, step=step)

    out_path = TABLES_DIR / "RQ5_Table2.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        if not bt.empty: bt.to_excel(w, index=False, sheet_name="Backtest")
        df_f = pd.DataFrame({"Date": series.index.append(fc.index)})
        df_f["Actual"] = series.reindex(df_f["Date"].values).values
        df_f["Forecast"] = fc.reindex(df_f["Date"].values).values
        df_f.to_excel(w, index=False, sheet_name="Forecast14")
    return out_path, ma, fc

def main():
    ensure_dirs()
    h = load_hdhi_clean()

    fig1 = make_los_distribution_figure(h)
    print(f"✅ Saved: {fig1}")

    series = build_occupancy_series(h)
    out_t2, ma, fc = forecast_and_accuracy(series, horizon=14, window=7, folds=4, step=14)
    print(f"✅ Saved: {out_t2}")

    fig2 = make_occupancy_figure(series, ma, fc)
    print(f"✅ Saved: {fig2}")

    t1 = los_summary_tables(h)
    print(f"✅ Saved: {t1}")

    print("\nRQ5 artifacts generated successfully.")
