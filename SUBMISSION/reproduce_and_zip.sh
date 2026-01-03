#!/usr/bin/env bash
set -euo pipefail
echo "Starting Airflow services..."
docker compose up -d webserver scheduler
echo "Triggering DAG..."
docker compose exec webserver airflow dags trigger hospital_project_pipeline
echo "Waiting for scheduler to pick up the run..."
sleep 15
echo "Latest runs:"
docker compose exec webserver airflow dags list-runs -d hospital_project_pipeline | tail -n 5
echo "Rebuilding ZIP from figures/ and tables/..."
zip -r Figures_and_Tables.zip figures tables
echo "Done. ZIP at: Figures_and_Tables.zip"
