import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.paths import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, DATA_DIR, ensure_dirs

# ---------------------------------------------------
# Helpers: date keys and safe conversions
# ---------------------------------------------------

def to_datekey(dt: pd.Series) -> pd.Series:
    """Convert a datetime series to yyyymmdd integer keys; NaT -> NaN."""
    return pd.to_datetime(dt, errors="coerce").dt.strftime("%Y%m%d").astype("Int64")

def dkey(dt: pd.Timestamp) -> int:
    return int(dt.strftime("%Y%m%d"))

# ---------------------------------------------------
# Build star schema (in memory) and save CSVs
# ---------------------------------------------------

def build_star_schema():
    # Load cleaned data
    k = pd.read_csv(PROCESSED_DIR / "kaggle_clean.csv")
    h = pd.read_csv(PROCESSED_DIR / "hdhi_clean.csv")

    # --- DimPatient ---
    # Kaggle patientid + HDHI mrd_no; make them strings to avoid float notation
    dim_patient_k = pd.DataFrame({
        "PatientKey": k["patientid"].astype(str).rename("PatientKey"),
        "Source": "Kaggle",
        "Age": k.get("age", np.nan),
        "Gender": k.get("gender", np.nan),
        "Neighbourhood": k.get("neighbourhood", np.nan)
    }).drop_duplicates(subset=["PatientKey"])

    dim_patient_h = pd.DataFrame({
        "PatientKey": h["mrd_no"].astype(str).rename("PatientKey"),
        "Source": "HDHI",
        "Age": h.get("age", np.nan),
        "Gender": h.get("gender", np.nan),
        "Neighbourhood": np.nan  # HDHI doesn't provide 'neighbourhood'
    }).drop_duplicates(subset=["PatientKey"])

    dim_patient = pd.concat([dim_patient_k, dim_patient_h], ignore_index=True).drop_duplicates(subset=["PatientKey"])

    # --- DimDate ---
    # Collect unique dates from Kaggle appointmentday/scheduledday and HDHI d_o_a/d_o_d
    dates = []
    for col in ["appointmentday", "scheduledday"]:
        if col in k.columns:
            dates.append(pd.to_datetime(k[col], errors="coerce"))
    for col in ["d_o_a", "d_o_d"]:
        if col in h.columns:
            dates.append(pd.to_datetime(h[col], errors="coerce"))
    dates = pd.to_datetime(pd.concat(dates), errors="coerce").dropna().dt.normalize().unique()
    dim_date = pd.DataFrame({"Date": dates})
    dim_date["DateKey"] = dim_date["Date"].dt.strftime("%Y%m%d").astype(int)
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Month"] = dim_date["Date"].dt.month
    dim_date["Day"] = dim_date["Date"].dt.day
    dim_date["Weekday"] = dim_date["Date"].dt.day_name()

    # --- DimDepartment ---
    # Use Kaggle 'neighbourhood' as a department-like dimension (location).
    if "neighbourhood" in k.columns:
        deps = pd.DataFrame({"DepartmentName": k["neighbourhood"].dropna().unique()})
    else:
        deps = pd.DataFrame({"DepartmentName": []})
    dims_department = deps.copy()
    dims_department["DepartmentID"] = np.arange(1, len(dims_department) + 1)
    dims_department["Category"] = "OutpatientLocation"

    # --- FactAppointments ---
    fact_appt = pd.DataFrame({
        "AppointmentID": k.get("appointmentid", np.nan),
        "PatientKey": k["patientid"].astype(str),
        "AppointmentDateKey": to_datekey(k.get("appointmentday", pd.Series(dtype="datetime64[ns]"))),
        "ScheduledDateKey": to_datekey(k.get("scheduledday", pd.Series(dtype="datetime64[ns]"))),
        "SMSReceived": k.get("sms_received", np.nan),
        "ShowStatus": k.get("no_show", np.nan),
        "WaitingDays": k.get("waitingdays", np.nan)
    })

    # --- FactAdmissions ---
    fact_adm = pd.DataFrame({
        "PatientKey": h["mrd_no"].astype(str),
        "AdmissionDateKey": to_datekey(h.get("d_o_a", pd.Series(dtype="datetime64[ns]"))),
        "LOS": pd.to_numeric(h.get("final_los", np.nan), errors="coerce"),
        "AdmissionType": h.get("type_of_admission_emergency_opd", np.nan)
    })

    # --- FactDischarges ---
    fact_dis = pd.DataFrame({
        "PatientKey": h["mrd_no"].astype(str),
        "DischargeDateKey": to_datekey(h.get("d_o_d", pd.Series(dtype="datetime64[ns]"))),
        "Outcome": h.get("outcome_norm", h.get("outcome", np.nan))
    })

    # --- FactDailyCensus (bed occupancy per day) ---
    # Efficient event-based method: +1 at admission, -1 at discharge; cumulative sum => occupancy.
    h["d_o_a_dt"] = pd.to_datetime(h.get("d_o_a"), errors="coerce").dt.normalize()
    h["d_o_d_dt"] = pd.to_datetime(h.get("d_o_d"), errors="coerce").dt.normalize()
    events = []
    # +1 at admission date
    add = h["d_o_a_dt"].dropna()
    events.append(pd.DataFrame({"Date": add, "delta": 1}))
    # -1 at discharge date
    sub = h["d_o_d_dt"].dropna()
    events.append(pd.DataFrame({"Date": sub, "delta": -1}))
    ev = pd.concat(events).groupby("Date", as_index=False)["delta"].sum().sort_values("Date")

    # Build full date range from min(admission) to max(discharge)
    if not ev.empty:
        full_range = pd.date_range(ev["Date"].min(), ev["Date"].max(), freq="D")
        df_occ = pd.DataFrame({"Date": full_range})
        df_occ = df_occ.merge(ev, on="Date", how="left").fillna({"delta": 0})
        df_occ["OccupiedBeds"] = df_occ["delta"].cumsum().astype(int)
    else:
        df_occ = pd.DataFrame(columns=["Date", "OccupiedBeds"])

    fact_census = pd.DataFrame({
        "DateKey": df_occ["Date"].dt.strftime("%Y%m%d").astype(int),
        "OccupiedBeds": df_occ["OccupiedBeds"]
    })

    # Save star pieces for inspection (optional)
    star_dir = DATA_DIR / "processed" / "star"
    star_dir.mkdir(parents=True, exist_ok=True)
    dim_patient.to_csv(star_dir / "DimPatient.csv", index=False)
    dim_date.to_csv(star_dir / "DimDate.csv", index=False)
    dims_department.to_csv(star_dir / "DimDepartment.csv", index=False)
    fact_appt.to_csv(star_dir / "FactAppointments.csv", index=False)
    fact_adm.to_csv(star_dir / "FactAdmissions.csv", index=False)
    fact_dis.to_csv(star_dir / "FactDischarges.csv", index=False)
    fact_census.to_csv(star_dir / "FactDailyCensus.csv", index=False)

    # Return frames for downstream use
    return {
        "DimPatient": dim_patient,
        "DimDate": dim_date,
        "DimDepartment": dims_department,
        "FactAppointments": fact_appt,
        "FactAdmissions": fact_adm,
        "FactDischarges": fact_dis,
        "FactDailyCensus": fact_census,
    }

# ---------------------------------------------------
# RQ3_Fig1: Star schema diagram
# ---------------------------------------------------

def make_star_schema_diagram():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Central facts
    ax.text(0.50, 0.65,
            "FactAppointments\n(AppointmentID, PatientKey,\nAppointmentDateKey, ScheduledDateKey,\nSMSReceived, ShowStatus, WaitingDays)",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="#fff2cc", ec="#bf9000"), fontsize=9)

    ax.text(0.50, 0.45,
            "FactAdmissions\n(PatientKey, AdmissionDateKey,\nLOS, AdmissionType)",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="#fff2cc", ec="#bf9000"), fontsize=9)

    ax.text(0.50, 0.25,
            "FactDischarges\n(PatientKey, DischargeDateKey,\nOutcome)",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="#fff2cc", ec="#bf9000"), fontsize=9)

    # Dimensions
    ax.text(0.15, 0.80, "DimPatient\n(PatientKey, Age, Gender,\nNeighbourhood, Source)",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="#d9ead3", ec="#38761d"), fontsize=9)
    ax.text(0.85, 0.80, "DimDate\n(DateKey, Day, Week, Month,\nYear, Weekday)",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="#d9ead3", ec="#38761d"), fontsize=9)
    ax.text(0.85, 0.10, "DimDepartment\n(DepartmentID, DepartmentName,\nCategory)",
            ha="center", va="center", bbox=dict(boxstyle="round", fc="#d9ead3", ec="#38761d"), fontsize=9)

    # Simple arrows (dimension -> fact)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#38761d"))

    for y in [0.65, 0.45, 0.25]:
        arrow(0.22, 0.80, 0.42, y)   # DimPatient -> Facts
        arrow(0.78, 0.80, 0.58, y)   # DimDate -> Facts
    arrow(0.78, 0.10, 0.58, 0.65)    # DimDepartment -> FactAppointments

    out_path = FIGURES_DIR / "RQ3_Fig1.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------
# RQ3_Fig2: Raw vs Star performance comparison
# ---------------------------------------------------

def benchmark_queries(frames):
    """
    Compare runtime of two queries:
      Q1: Daily occupancy series (raw vs. precomputed FactDailyCensus)
      Q2: Average LOS by admission type (raw vs. FactAdmissions)
    """
    # Load raw HDHI for 'raw' paths
    h = pd.read_csv(PROCESSED_DIR / "hdhi_clean.csv")
    h["d_o_a_dt"] = pd.to_datetime(h.get("d_o_a"), errors="coerce").dt.normalize()
    h["d_o_d_dt"] = pd.to_datetime(h.get("d_o_d"), errors="coerce").dt.normalize()

    # --- Q1 Raw: occupancy via events from raw table ---
    t0 = time.time()
    events = []
    add = h["d_o_a_dt"].dropna()
    events.append(pd.DataFrame({"Date": add, "delta": 1}))
    sub = h["d_o_d_dt"].dropna()
    events.append(pd.DataFrame({"Date": sub, "delta": -1}))
    ev = pd.concat(events).groupby("Date", as_index=False)["delta"].sum().sort_values("Date")
    if not ev.empty:
        full_range = pd.date_range(ev["Date"].min(), ev["Date"].max(), freq="D")
        df_occ_raw = pd.DataFrame({"Date": full_range})
        df_occ_raw = df_occ_raw.merge(ev, on="Date", how="left").fillna({"delta": 0})
        df_occ_raw["OccupiedBeds"] = df_occ_raw["delta"].cumsum().astype(int)
    else:
        df_occ_raw = pd.DataFrame(columns=["Date", "OccupiedBeds"])
    t1 = time.time()
    q1_raw_sec = t1 - t0

    # --- Q1 Star: use FactDailyCensus directly ---
    t2 = time.time()
    df_occ_star = frames["FactDailyCensus"].copy()
    # Simple aggregation to emulate a query (e.g., monthly average occupancy)
    df_occ_star["YearMonth"] = df_occ_star["DateKey"].astype(str).str.slice(0, 6)
    _ = df_occ_star.groupby("YearMonth", as_index=False)["OccupiedBeds"].mean()
    t3 = time.time()
    q1_star_sec = t3 - t2

    # --- Q2 Raw: LOS by admission type from raw HDHI ---
    t4 = time.time()
    # Use final_los and type_of_admission_emergency_opd
    h["final_los"] = pd.to_numeric(h.get("final_los"), errors="coerce")
    q2_raw = h.groupby("type_of_admission_emergency_opd", as_index=False)["final_los"].mean()
    t5 = time.time()
    q2_raw_sec = t5 - t4

    # --- Q2 Star: LOS by admission type from FactAdmissions ---
    t6 = time.time()
    fa = frames["FactAdmissions"]
    q2_star = fa.groupby("AdmissionType", as_index=False)["LOS"].mean()
    t7 = time.time()
    q2_star_sec = t7 - t6

    return {
        "Q1_raw_sec": q1_raw_sec,
        "Q1_star_sec": q1_star_sec,
        "Q2_raw_sec": q2_raw_sec,
        "Q2_star_sec": q2_star_sec
    }

def make_perf_figure(timings):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Occupancy (raw)", "Occupancy (star)", "Avg LOS (raw)", "Avg LOS (star)"]
    values = [timings["Q1_raw_sec"], timings["Q1_star_sec"], timings["Q2_raw_sec"], timings["Q2_star_sec"]]
    colors = ["#cc0000", "#6aa84f", "#cc0000", "#6aa84f"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.02, f"{v:.3f}s", ha="center")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Raw vs Star Query Performance (two representative queries)")
    plt.xticks(rotation=20)
    out_path = FIGURES_DIR / "RQ3_Fig2.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------
# RQ3 tables: schema catalog & KPI mapping
# ---------------------------------------------------

def make_schema_catalog(frames):
    rows = []
    # Table name, type, primary key, grain, key columns (human-readable)
    rows += [["DimPatient", "Dimension", "PatientKey", "One row per patient",
              "PatientKey; Age; Gender; Neighbourhood; Source"]]
    rows += [["DimDate", "Dimension", "DateKey", "One row per date",
              "DateKey; Year; Month; Day; Weekday"]]
    rows += [["DimDepartment", "Dimension", "DepartmentID", "One row per department/location",
              "DepartmentID; DepartmentName; Category"]]
    rows += [["FactAppointments", "Fact", "AppointmentID", "One row per appointment",
              "AppointmentID; PatientKey; AppointmentDateKey; ScheduledDateKey; SMSReceived; ShowStatus; WaitingDays"]]
    rows += [["FactAdmissions", "Fact", "(PatientKey, AdmissionDateKey)", "One row per admission event",
              "PatientKey; AdmissionDateKey; LOS; AdmissionType"]]
    rows += [["FactDischarges", "Fact", "(PatientKey, DischargeDateKey)", "One row per discharge event",
              "PatientKey; DischargeDateKey; Outcome"]]
    rows += [["FactDailyCensus", "Fact", "DateKey", "One row per day (hospital-level)",
              "DateKey; OccupiedBeds"]]

    df = pd.DataFrame(rows, columns=["TableName", "Type", "PrimaryKey", "Grain", "KeyColumns"])
    out_path = TABLES_DIR / "RQ3_Table1.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="SchemaCatalog")
    return out_path

def make_kpi_mapping():
    rows = []
    rows += [["Bed occupancy rate", "FactDailyCensus", "DimDate", "Average occupied beds per day/period"]]
    rows += [["Average LOS", "FactAdmissions", "DimDate, DimPatient (optional), AdmissionType",
              "Mean LOS across admissions (days)"]]
    rows += [["Admission volume", "FactAdmissions", "DimDate, AdmissionType",
              "Count of admissions per day/period"]]
    rows += [["Discharge rate", "FactDischarges", "DimDate",
              "Count of discharges per day/period"]]
    rows += [["No-show rate", "FactAppointments", "DimDate, DimDepartment (neighbourhood)",
              "Share of appointments not attended"]]

    df = pd.DataFrame(rows, columns=["MetricName", "FactTable", "DimensionsUsed", "Description"])
    out_path = TABLES_DIR / "RQ3_Table2.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="KPIs")
    return out_path

# ---------------------------------------------------
# Driver
# ---------------------------------------------------

def main():
    ensure_dirs()
    frames = build_star_schema()

    fig1 = make_star_schema_diagram()
    print(f"✅ Saved: {fig1}")

    timings = benchmark_queries(frames)
    fig2 = make_perf_figure(timings)
    print(f"✅ Saved: {fig2} (timings: {timings})")

    t1 = make_schema_catalog(frames)
    print(f"✅ Saved: {t1}")

    t2 = make_kpi_mapping()
    print(f"✅ Saved: {t2}")

    print("\nRQ3 artifacts generated successfully.")
