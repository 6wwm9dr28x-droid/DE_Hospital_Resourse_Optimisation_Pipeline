import os, time, uuid, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs

# ------------------------------
# RQ4_Fig1: DAG / block diagram
# ------------------------------

def make_dag_diagram():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    boxes = [
        ("extract_data\n(read raw CSVs)",        0.08, 0.80),
        ("clean_data\n(parse dates, dedup)",     0.26, 0.80),
        ("transform_features\n(waitingdays, LOS)",0.44, 0.80),
        ("build_star_schema\n(dims + facts)",    0.62, 0.80),
        ("train_no_show_model\n(RQ2)",           0.80, 0.80),
        ("model_los_and_forecast\n(RQ5)",        0.44, 0.54),
        ("generate_outputs\n(PDFs/XLSX/CSV)",    0.62, 0.54),
        ("write_provenance\n(hashes, run_id)",   0.80, 0.54),
    ]
    for txt, x, y in boxes:
        ax.text(x, y, txt, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="#e8f0fe", ec="#3366cc"), fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#3366cc"))

    arrow(0.16, 0.80, 0.22, 0.80)  # extract -> clean
    arrow(0.34, 0.80, 0.40, 0.80)  # clean -> transform
    arrow(0.52, 0.80, 0.58, 0.80)  # transform -> star
    arrow(0.70, 0.80, 0.76, 0.80)  # star -> no_show
    arrow(0.52, 0.74, 0.52, 0.58)  # transform -> RQ5
    arrow(0.70, 0.74, 0.68, 0.58)  # star -> outputs
    arrow(0.76, 0.74, 0.78, 0.58)  # star -> provenance

    out_path = FIGURES_DIR / "RQ4_Fig1.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path

# -------------------------------------------
# RQ4 runtime benchmark & plot (Kaggle only)
# -------------------------------------------

def _timed_clean_like_ops(df: pd.DataFrame) -> float:
    """
    Simulate cleaning-like operations on a sample:
    parse dates -> compute waitingdays -> drop duplicates.
    Returns elapsed seconds.
    """
    t0 = time.time()
    df2 = df.copy()
    # parse to datetime (coerce)
    for col in ["scheduledday", "appointmentday"]:
        if col in df2.columns:
            df2[col] = pd.to_datetime(df2[col], errors="coerce")
    # waiting days
    if {"appointmentday", "scheduledday"}.issubset(df2.columns):
        df2["waitingdays"] = (df2["appointmentday"] - df2["scheduledday"]).dt.days
    # dedup by keys when present
    subset = [c for c in ["patientid", "appointmentid"] if c in df2.columns]
    df2 = df2.drop_duplicates(subset=subset) if subset else df2.drop_duplicates()
    return time.time() - t0

def make_runtime_benchmarks():
    kag = pd.read_csv(PROCESSED_DIR / "kaggle_clean.csv")
    sizes = [5000, 20000, 50000, 100000]
    rows = []
    for n in sizes:
        m = min(n, len(kag))
        sample = kag.sample(n=m, random_state=42)
        secs = _timed_clean_like_ops(sample)
        thr = m / secs if secs > 0 else np.nan
        rows.append({"Rows": m, "Runtime_sec": round(secs, 4), "Throughput_rows_per_sec": round(thr, 2)})
    df = pd.DataFrame(rows)
    # Save table
    out_csv = TABLES_DIR / "RQ4_Table1.csv"
    df.to_csv(out_csv, index=False)
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["Rows"], df["Runtime_sec"], marker="o")
    ax.set_title("Pipeline Cleaning Runtime vs Rows (sampled Kaggle)")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Runtime (seconds)")
    for x, y in zip(df["Rows"], df["Runtime_sec"]):
        ax.text(x, y, f"{y:.3f}s", ha="left", va="bottom")
    out_fig = FIGURES_DIR / "RQ4_Fig2.pdf"
    fig.savefig(out_fig, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_csv, out_fig

# -------------------------------------------
# RQ4 reproducibility report (hashes, run_id)
# -------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def make_reproducibility_report():
    run_id = f"run_{uuid.uuid4().hex[:8]}_{time.strftime('%Y%m%d_%H%M%S')}"
    figs = sorted([p for p in FIGURES_DIR.glob("RQ*_Fig*.pdf")])
    tabs = sorted([p for p in TABLES_DIR.glob("RQ*_Table*.xlsx")] + [p for p in TABLES_DIR.glob("RQ*_Table*.csv")])

    rows = []
    for p in figs + tabs:
        stat = p.stat()
        rows.append({
            "run_id": run_id,
            "file": p.name,
            "path": str(p),
            "sha256": _sha256(p),
            "size_bytes": stat.st_size,
            "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        })
    df = pd.DataFrame(rows)
    out_path = TABLES_DIR / "RQ4_Table2.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Reproducibility")
    return out_path, run_id

# -----------------------------
# Driver
# -----------------------------

def main():
    ensure_dirs()
    f1 = make_dag_diagram()
    print(f"✅ Saved: {f1}")

    t1, f2 = make_runtime_benchmarks()
    print(f"✅ Saved: {t1}")
    print(f"✅ Saved: {f2}")

    t2, run_id = make_reproducibility_report()
    print(f"✅ Saved: {t2} (run_id={run_id})")

    print("\nRQ4 artifacts generated successfully.")
