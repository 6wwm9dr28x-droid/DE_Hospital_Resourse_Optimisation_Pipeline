import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.paths import (
    INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs
)

# ---------------------------
# Helpers
# ---------------------------

def _avg_missing(df: pd.DataFrame) -> float:
    """Average missingness in percent."""
    if df.empty: return 0.0
    return float(df.isna().mean().mean() * 100.0)

def _dup_count_kaggle_raw(df: pd.DataFrame) -> int:
    """
    Count duplicates in raw Kaggle using keys if present (PatientId + AppointmentID),
    otherwise count full-row duplicates.
    """
    cols = df.columns
    key_subset = []
    if "PatientId" in cols:      key_subset.append("PatientId")
    if "AppointmentID" in cols:  key_subset.append("AppointmentID")
    if key_subset:
        return int(df.duplicated(subset=key_subset).sum())
    return int(df.duplicated().sum())

def _dup_count_hdhi_raw(df: pd.DataFrame) -> int:
    """For HDHI raw, count full-row duplicates (no reliable key combo in raw)."""
    return int(df.duplicated().sum())

# ---------------------------
# RQ1_Fig1: Pipeline diagram
# ---------------------------

def make_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    boxes = [
        ("Extract CSVs\n(Kaggle Appointments, HDHI Admissions)", 0.08, 0.80),
        ("Clean & Validate\n(date parsing, dedup, type coercion)", 0.36, 0.80),
        ("Transform Features\n(waiting days, LOS, outcome codes)", 0.64, 0.80),
        ("Dimensional Model\n(DimPatient, DimDate, DimDepartment)", 0.36, 0.52),
        ("Facts\n(FactAppointments, FactAdmissions, FactDischarges)", 0.64, 0.52),
        ("Analytics & Outputs\n(figures & tables for RQs)", 0.50, 0.25),
    ]
    for txt, x, y in boxes:
        ax.text(
            x, y, txt, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#e8f0fe", ec="#3366cc"),
            fontsize=10
        )

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#3366cc"))

    arrow(0.19, 0.80, 0.31, 0.80)
    arrow(0.47, 0.80, 0.59, 0.80)
    arrow(0.47, 0.74, 0.47, 0.56)
    arrow(0.59, 0.74, 0.59, 0.56)
    arrow(0.47, 0.46, 0.50, 0.29)

    out_path = FIGURES_DIR / "RQ1_Fig1.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path

# ----------------------------------------
# RQ1_Fig2: Missingness heatmap (2x2 grid)
# ----------------------------------------

def make_missingness_heatmap():
    # Load raw & cleaned
    k_raw = pd.read_csv(INTERIM_DIR / "kaggle_raw.csv")
    h_raw = pd.read_csv(INTERIM_DIR / "hdhi_raw.csv")
    k = pd.read_csv(PROCESSED_DIR / "kaggle_clean.csv")
    h = pd.read_csv(PROCESSED_DIR / "hdhi_clean.csv")

    data = np.array([
        [_avg_missing(k_raw), _avg_missing(k)],
        [_avg_missing(h_raw), _avg_missing(h)]
    ])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(data, cmap="Reds", vmin=0, vmax=max(1.0, data.max()))

    ax.set_xticks([0, 1], labels=["Before ETL", "After ETL"])
    ax.set_yticks([0, 1], labels=["Kaggle", "HDHI"])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}%", ha="center", va="center", color="black")

    ax.set_title("Average Missing Values (%) — Before vs After ETL")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_path = FIGURES_DIR / "RQ1_Fig2.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path

# ----------------------------------------------------
# RQ1_Table1: Field-to-model mapping (Kaggle & HDHI)
# ----------------------------------------------------

def make_mapping_table():
    # Read cleaned datasets (standardized headers)
    k = pd.read_csv(PROCESSED_DIR / "kaggle_clean.csv")
    h = pd.read_csv(PROCESSED_DIR / "hdhi_clean.csv")

    rows = []

    # Kaggle mapping (standardized names)
    k_map = {
        "patientid":                  ("DimPatient.PatientKey",      "Patient surrogate key"),
        "appointmentid":              ("FactAppointments.AppointmentID", "Appointment record id"),
        "gender":                     ("DimPatient.Gender",          "Patient gender"),
        "scheduledday":               ("FactAppointments.ScheduledDateKey",  "Scheduled timestamp"),
        "appointmentday":             ("FactAppointments.AppointmentDateKey","Appointment date"),
        "age":                        ("DimPatient.Age",             "Patient age"),
        "neighbourhood":              ("DimDepartment.DepartmentName","Clinic location / neighbourhood"),
        "scholarship":                ("DimPatient.Scholarship",     "Social program flag"),
        "hypertension":               ("DimPatient.Hypertension",    "Comorbidity"),
        "diabetes":                   ("DimPatient.Diabetes",        "Comorbidity"),
        "alcoholism":                 ("DimPatient.Alcoholism",      "Comorbidity"),
        "handcap":                    ("DimPatient.Handicap",        "Disability flag"),
        "sms_received":               ("FactAppointments.SMSReceived","Reminder message flag"),
        "no_show":                    ("FactAppointments.ShowStatus", "Attendance outcome"),
        "waitingdays":                ("FactAppointments.WaitingDays","Derived feature: days between scheduling and appointment"),
    }
    for col, (target, semantics) in k_map.items():
        if col in k.columns:
            rows.append(["Kaggle", col, semantics, target])

    # HDHI mapping (standardized names)
    h_map = {
        "mrd_no":                     ("DimPatient.PatientKey",      "Medical record number"),
        "d_o_a":                      ("FactAdmissions.AdmissionDateKey","Date of admission"),
        "d_o_d":                      ("FactDischarges.DischargeDateKey","Date of discharge"),
        "age":                        ("DimPatient.Age",             "Patient age"),
        "gender":                     ("DimPatient.Gender",          "Patient gender"),
        "type_of_admission_emergency_opd": ("FactAdmissions.AdmissionType","Admission type"),
        "duration_of_stay":           ("FactAdmissions.LOS",         "Length of stay (days) raw"),
        "final_los":                  ("FactAdmissions.LOS",         "Length of stay (days) cleaned"),
        "outcome":                    ("FactDischarges.Outcome",     "Discharge outcome raw"),
        "outcome_norm":               ("FactDischarges.Outcome",     "Discharge outcome normalized"),
    }
    for col, (target, semantics) in h_map.items():
        if col in h.columns:
            rows.append(["HDHI", col, semantics, target])

    df = pd.DataFrame(rows, columns=["Source", "Field", "Semantics", "TargetModel"])

    out_path = TABLES_DIR / "RQ1_Table1.xlsx"
    # Write to Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="FieldMapping")
    return out_path

# ------------------------------------------------
# RQ1_Table2: Data quality audit (raw vs cleaned)
# ------------------------------------------------

def make_quality_table():
    k_raw = pd.read_csv(INTERIM_DIR / "kaggle_raw.csv")
    h_raw = pd.read_csv(INTERIM_DIR / "hdhi_raw.csv")
    k = pd.read_csv(PROCESSED_DIR / "kaggle_clean.csv")
    h = pd.read_csv(PROCESSED_DIR / "hdhi_clean.csv")

    rows = []

    # Kaggle metrics
    k_dups_raw = _dup_count_kaggle_raw(k_raw)
    k_dups_clean = int(k.duplicated().sum())  # After cleaning (we removed key-based dupes)
    rows += [
        ["Kaggle", "Rows_Raw", len(k_raw)],
        ["Kaggle", "Rows_Clean", len(k)],
        ["Kaggle", "Duplicates_Raw", k_dups_raw],
        ["Kaggle", "Duplicates_Clean", k_dups_clean],
        ["Kaggle", "AvgMissing_BeforePct", _avg_missing(k_raw)],
        ["Kaggle", "AvgMissing_AfterPct", _avg_missing(k)],
    ]

    # HDHI metrics
    h_dups_raw = _dup_count_hdhi_raw(h_raw)
    h_dups_clean = int(h.duplicated().sum())
    rows += [
        ["HDHI", "Rows_Raw", len(h_raw)],
        ["HDHI", "Rows_Clean", len(h)],
        ["HDHI", "Duplicates_Raw", h_dups_raw],
        ["HDHI", "Duplicates_Clean", h_dups_clean],
        ["HDHI", "AvgMissing_BeforePct", _avg_missing(h_raw)],
        ["HDHI", "AvgMissing_AfterPct", _avg_missing(h)],
    ]

    df = pd.DataFrame(rows, columns=["Dataset", "Metric", "Value"])
    out_path = TABLES_DIR / "RQ1_Table2.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="QualityAudit")
    return out_path

# ---------------------------
# Driver
# ---------------------------

def main():
    ensure_dirs()

    fig1 = make_pipeline_diagram()
    print(f"✅ Saved: {fig1}")

    fig2 = make_missingness_heatmap()
    print(f"✅ Saved: {fig2}")

    t1 = make_mapping_table()
    print(f"✅ Saved: {t1}")

    t2 = make_quality_table()
    print(f"✅ Saved: {t2}")

    print("\nRQ1 artifacts generated successfully.")

if __name__ == "__main__":
    main()
