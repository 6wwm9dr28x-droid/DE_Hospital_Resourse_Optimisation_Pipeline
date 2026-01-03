# Hospital Resource Optimization Pipeline (Data Engineering)

This project implements a reproducible **data‑engineering pipeline** integrating:
- **Outpatient appointments** — `hospital-KaggleV2-May-2016.csv`
- **Inpatient admissions** — `HDHI Admission data.csv`

The pipeline performs **ingestion → cleaning → feature derivation → star schema build → artifact generation** and is orchestrated by an **Airflow DAG**: `hospital_project_pipeline`.

---

## Deliverables (all code‑generated)

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
3. Trigger the DAG:
   ```bash
   docker compose exec webserver airflow dags trigger hospital_project_pipeline
### B) Local Python (no Airflow)
## Rebuild the submission ZIP 
zip -r Figures_and_Tables.zip figures tables
