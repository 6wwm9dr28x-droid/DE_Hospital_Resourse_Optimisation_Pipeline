from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# --- Task callables import your modules and call their 'main' functions ---
def task_extract_data():
    from src.data_ingestion.ingest import main as ingest_main
    ingest_main()

def task_clean_data():
    from src.data_cleaning.clean import main as clean_main
    clean_main()

def task_build_star_schema():
    from src.evaluation.rq3_outputs import build_star_schema
    _ = build_star_schema()

def task_generate_rq1():
    from src.evaluation.rq1_outputs import main as rq1_main
    rq1_main()

def task_generate_rq2():
    from src.modeling.no_show import main as rq2_main
    rq2_main()

def task_generate_rq3():
    from src.evaluation.rq3_outputs import main as rq3_main
    rq3_main()

def task_generate_rq4():
    from src.evaluation.rq4_outputs import main as rq4_main
    rq4_main()

def task_generate_rq5():
    from src.modeling.los_forecast import main as rq5_main
    rq5_main()

def task_write_provenance():
    from src.evaluation.rq4_outputs import make_reproducibility_report
    _path, _run_id = make_reproducibility_report()

default_args = {"owner": "data-eng-student", "retries": 0}

with DAG(
    dag_id="hospital_project_pipeline",
    start_date=datetime(2025, 12, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    tags=["hospital", "data_engineering"],
    description="Hospital Resource Optimization Pipeline (ETL + RQ1–RQ5 artifacts)"
) as dag:
    t1 = PythonOperator(task_id="extract_data",          python_callable=task_extract_data)
    t2 = PythonOperator(task_id="clean_data",            python_callable=task_clean_data)
    t3 = PythonOperator(task_id="build_star_schema",     python_callable=task_build_star_schema)
    t4 = PythonOperator(task_id="generate_rq1",          python_callable=task_generate_rq1)
    t5 = PythonOperator(task_id="generate_rq2",          python_callable=task_generate_rq2)
    t6 = PythonOperator(task_id="generate_rq3",          python_callable=task_generate_rq3)
    t7 = PythonOperator(task_id="generate_rq4",          python_callable=task_generate_rq4)
    t8 = PythonOperator(task_id="generate_rq5",          python_callable=task_generate_rq5)
    t9 = PythonOperator(task_id="write_provenance",      python_callable=task_write_provenance)

    t1 >> t2 >> t3
    t3 >> [t4, t5, t6, t7, t8] >> t9
