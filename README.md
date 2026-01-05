# Hospital Resource Optimization Pipeline (Data Engineering)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Deliverables](#deliverables)
- [Team & Roles](#team--roles)
- [Project Structure](#project-structure)
- [Datasets](#Datasets)
- [Reproduce the Results](#reproduce-the-results)

---

## Project Overview
This project implements a reproducible **data‑engineering pipeline** integrating:
- **Outpatient appointments** — `hospital-KaggleV2-May-2016.csv`
- **Inpatient admissions** — `HDHI Admission data.csv`

The pipeline performs **ingestion → cleaning → feature derivation → star schema build → artifact generation** and is orchestrated by an **Airflow DAG**: `hospital_project_pipeline`.

---

## Deliverables 

- **RQ1 – Integration & Quality**  
  **Figures:** `RQ1_Fig1.pdf` (pipeline), `RQ1_Fig2.pdf` (missingness)  
  **Tables:** `RQ1_Table1.xlsx` (field→model mapping), `RQ1_Table2.xlsx` (quality audit)

- **RQ2 – No‑show Modeling (logistic)**  
  **Figures:** `RQ2_Fig1.pdf` (WaitingDays vs outcome), `RQ2_Fig2.pdf` (ROC + confusion matrix)  
  **Tables:** `RQ2_Table1.xlsx` (coefficients/OR/metrics), `RQ2_Table2.xlsx` (no‑show by cohort)

- **RQ3 – Star Schema & Performance**  
  **Figures:** `RQ3_Fig1.pdf` (schema), `RQ3_Fig2.pdf` (raw vs star timings)  
  **Tables:** `RQ3_Table1.xlsx` (schema catalog), `RQ3_Table2.xlsx` (KPI mapping)

- **RQ4 – Scalability & Reproducibility**  
  **Figures:** `RQ4_Fig1.pdf` (DAG), `RQ4_Fig2.pdf` (runtime vs rows)  
  **Tables:** `RQ4_Table1.csv` (benchmark), `RQ4_Table2.xlsx` (artifact SHA‑256 with run_id)

- **RQ5 – LOS & Occupancy Forecasting**  
  **Figures:** `RQ5_Fig1.pdf` (LOS distributions), `RQ5_Fig2.pdf` (occupancy + 14‑day MA forecast)  
  **Tables:** `RQ5_Table1.xlsx` (LOS summaries), `RQ5_Table2.xlsx` (backtest + forecast series)

#### All tables and figures are code generated
---

## Project Structure

   ```bash
 dags/                  # Airflow DAG definition
 src/                   # Core pipeline logic
 data/                  # Intermediate and processed data (no raw uploads)
 figures/               # Auto-generated PDF figures
 tables/                # Auto-generated tables (CSV/XLSX)
 SUBMISSION/            # Submission templates and instructions
   ```

---

## Datasets

### 1. Outpatient Appointments (No-Show Analysis)

- **Source:** Kaggle – Medical Appointment No Shows Dataset
- **Link** [Kaggle – Medical Appointment No Shows] [([https://www.kaggle.com/datasets/joniarroba/hospital-mortality](https://www.kaggle.com/datasets/muhammetgamal5/noshowappointmentskagglev2may2016csv?utm_source=copilot.com))](https://www.kaggle.com/datasets/muhammetgamal5/noshowappointmentskagglev2may2016csv?utm_source=copilot.com)
- **Description:**  
  Contains anonymized outpatient appointment records, including patient demographics, scheduled dates, and no-show indicators.  
  Used to analyze outpatient appointment patterns and no-show behavior.

---

### 2. Inpatient Admissions (Resource Utilization)

- **Source:** HDHI –  Hospital Admission Dataset
- **Link** - **Inpatient admissions dataset:** [[Kaggle – German Hospital Admission Dataset](https://www.kaggle.com/datasets/saadfarooq1/german-hospital-admission)](https://www.kaggle.com/datasets/ruckdent/hdhi-admission-dataset?resource=download)
- **Description:**  
  Contains inpatient hospital admission records with diagnostic and administrative information.  
  Used to analyze inpatient load and resource utilization trends.

## Team & Roles

- **Magomed Makhsudov** — Technical Lead  
  (Pipeline implementation, Airflow DAG, experiments, figures & tables)

- **Uzoma Nnaemeka Eze** — Documentation & Presentation Lead  
  (Final report, slides, formatting, consistency)

---

## Latest successful run (proof of reproducibility)

- **Airflow DAG run_id:** `manual__2026-01-03T10:17:47+00:00` → **success**  
- **Provenance (artifact) run_id:** `run_6de73861_20260103_103008`  
  → See `tables/RQ4_Table2.xlsx` for file list, SHA‑256 hashes, sizes, and timestamps.

---

## Reproduce the results

### A) Docker (recommended)
1. Start services:
   ```bash
   docker compose up -d webserver scheduler

2. Trigger the DAG:
   ```bash
   docker compose exec webserver airflow dags trigger hospital_project_pipeline
### B) Local Python (no Airflow)
   ```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run pipeline step-by-step
python run_step2_ingest_clean.py
python run_step3_star_schema.py
python run_step4_generate_rq_outputs.py
   ```

### C) Rebuild the submission ZIP (after pipeline completion)
```bash
zip -r Figures_and_Tables.zip figures tables
```
---
