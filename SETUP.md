# Setup Guide

This guide provides step-by-step instructions to set up and run the Airflow ETL pipeline.

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 2.0 or higher)
- At least 4GB of free RAM
- At least 10GB of free disk space

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd ids_706
```

### 2. Create Environment File

Create a `.env` file in the project root:

```bash
cat > .env << EOF
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
EOF
```

Or on Windows (PowerShell):

```powershell
@"
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=airflow
_AIRFLOW_WWW_USER_PASSWORD=airflow
"@ | Out-File -FilePath .env -Encoding ASCII
```

### 3. Initialize Airflow Database

```bash
docker-compose up airflow-init
```

Wait until you see "airflow-init exited with code 0"

### 4. Start Airflow Services

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database
- Airflow webserver (http://localhost:8080)
- Airflow scheduler

### 5. Verify Services are Running

```bash
docker-compose ps
```

All services should show as "healthy" or "running"

### 6. Access Airflow Web UI

1. Open your browser and navigate to: http://localhost:8080
2. Login with:
   - Username: `airflow`
   - Password: `airflow`

### 7. Enable and Trigger the DAG

1. In the Airflow UI, find `etl_pipeline_dag`
2. Toggle the switch on the left to unpause the DAG
3. Click the play button (▶) to trigger a manual run
4. Or wait for the scheduled daily run

### 8. Monitor Execution

1. Click on the DAG name to see details
2. Switch to "Graph" view to see task dependencies
3. Click on individual tasks to see logs
4. Watch as tasks turn green upon successful completion

## Verification Steps

### Check Database Connection

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U airflow

# Inside psql, run:
\dt

# You should see the customer_orders table after DAG execution
SELECT COUNT(*) FROM customer_orders;

# Exit
\q
```

### Check Data Files

```bash
# List data files
docker-compose exec airflow-scheduler ls -lh /opt/airflow/data/

# View analysis results
docker-compose exec airflow-scheduler cat /opt/airflow/data/analysis_results.json

# View model results
docker-compose exec airflow-scheduler cat /opt/airflow/data/model_results.json
```

### Check Logs

```bash
# Scheduler logs
docker-compose logs airflow-scheduler

# Webserver logs
docker-compose logs airflow-webserver

# Follow logs in real-time
docker-compose logs -f airflow-scheduler
```

## Troubleshooting

### Problem: Port 8080 is already in use

Solution: Change the port in docker-compose.yml:

```yaml
airflow-webserver:
  ports:
    - "8081:8080"  # Change 8080 to 8081
```

### Problem: Permission denied errors

Solution: Set the correct AIRFLOW_UID:

```bash
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

### Problem: Services not starting

Solution: Reset everything:

```bash
docker-compose down -v
docker-compose up airflow-init
docker-compose up -d
```

### Problem: DAG not appearing

1. Check DAG file syntax:
```bash
docker-compose exec airflow-scheduler python /opt/airflow/dags/etl_pipeline_dag.py
```

2. Check scheduler logs:
```bash
docker-compose logs airflow-scheduler | grep etl_pipeline
```

### Problem: Tasks failing

1. Click on the failed task in the UI
2. Check the logs
3. Common issues:
   - Insufficient memory (allocate more RAM to Docker)
   - Missing dependencies (check requirements.txt)
   - Database connection issues (verify PostgreSQL is running)

## Stopping the Pipeline

### Stop services but keep data:
```bash
docker-compose down
```

### Stop services and remove all data:
```bash
docker-compose down -v
```

## Restarting

After stopping, restart with:

```bash
docker-compose up -d
```

No need to run `airflow-init` again unless you removed volumes.

## Performance Tips

1. **Allocate sufficient resources to Docker**:
   - Minimum: 4GB RAM, 2 CPUs
   - Recommended: 8GB RAM, 4 CPUs

2. **Monitor resource usage**:
```bash
docker stats
```

3. **Clean up old logs periodically**:
```bash
docker-compose exec airflow-scheduler airflow db clean --clean-before-timestamp "2024-01-01"
```

## Development Tips

### Run DAG tests locally

```bash
# Enter the scheduler container
docker-compose exec airflow-scheduler bash

# Test DAG loading
python /opt/airflow/dags/etl_pipeline_dag.py

# Test specific task
airflow tasks test etl_pipeline_dag ingest_customers 2024-01-01
```

### Modify the DAG

1. Edit `dags/etl_pipeline_dag.py`
2. Save the file
3. Wait 30 seconds for Airflow to detect changes
4. Refresh the UI

No need to restart services for DAG changes!

## Next Steps

After successful setup:

1. ✅ Take screenshots of the Graph view
2. ✅ Take screenshots of successful execution
3. ✅ Save screenshots to the `screenshots/` directory
4. ✅ Query the PostgreSQL database to verify data
5. ✅ Review analysis and ML results in the data directory

## Support

For issues or questions:
- Check the logs first
- Review the troubleshooting section
- Consult the Airflow documentation: https://airflow.apache.org/docs/

## Clean Shutdown

When you're done:

```bash
# Stop all services
docker-compose down

# Optional: Remove all data and volumes
docker-compose down -v

# Optional: Remove Docker images
docker-compose down --rmi all -v
```

---

**Happy Data Engineering! 🚀**
