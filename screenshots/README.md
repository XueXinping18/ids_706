# Screenshots

This directory contains screenshots demonstrating the successful execution of the Airflow ETL pipeline.

## Required Screenshots

1. **dag_graph.png** - DAG graph view showing task dependencies
2. **dag_success.png** - Successful DAG execution in the Airflow UI

## How to Capture Screenshots

1. Start the Airflow services: `docker-compose up -d`
2. Access Airflow UI at http://localhost:8080
3. Login with credentials (airflow/airflow)
4. Navigate to the `etl_pipeline_dag`
5. For Graph View:
   - Click on "Graph" tab
   - Take screenshot showing all tasks and their dependencies
6. For Successful Execution:
   - Trigger the DAG manually or wait for scheduled run
   - Take screenshot showing all tasks in green (success state)
   - Can be from Grid view or Graph view

Add your screenshots here after running the pipeline!
