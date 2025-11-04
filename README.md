# Airflow ETL Pipeline Project

## Overview

This project demonstrates a complete ETL (Extract, Transform, Load) pipeline using Apache Airflow with parallel processing capabilities. The pipeline ingests customer and order data, transforms them, merges the datasets, loads them into PostgreSQL, and performs data analysis including machine learning model training.

## Features

- **Parallel Processing**: Ingestion and transformation tasks run in parallel to optimize execution time
- **TaskGroups**: Related transformation tasks are organized using Airflow TaskGroups
- **Data Transformation**: Validation, enrichment, and metric calculation
- **Database Integration**: PostgreSQL for data storage
- **Analysis**: Statistical analysis and machine learning model training
- **Automated Cleanup**: Removes intermediate files after processing
- **Scheduled Execution**: Configured to run daily
- **Containerized Setup**: Complete Docker-based development environment

## Architecture

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Ingestion                          │
│                         (Parallel)                              │
├──────────────────────────────┬──────────────────────────────────┤
│    Ingest Customer Data      │      Ingest Order Data           │
└──────────────┬───────────────┴────────────────┬─────────────────┘
               │                                │
               ▼                                ▼
┌──────────────────────────────┐  ┌────────────────────────────────┐
│  Customer Transformation     │  │   Order Transformation         │
│  (TaskGroup)                 │  │   (TaskGroup)                  │
│  ┌────────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │ Validate Customers     │  │  │  │ Validate Orders          │ │
│  └───────────┬────────────┘  │  │  └──────────┬───────────────┘ │
│              ▼               │  │             ▼                 │
│  ┌────────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │ Enrich Customers       │  │  │  │ Calculate Metrics        │ │
│  └────────────────────────┘  │  │  └──────────────────────────┘ │
└──────────────┬───────────────┘  └────────────┬───────────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │    Merge Datasets    │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Load to PostgreSQL  │
                    └──────────┬───────────┘
                               │
               ┌───────────────┴────────────────┐
               │                                │
               ▼                                ▼
    ┌──────────────────┐          ┌────────────────────────┐
    │  Perform Analysis│          │  Train ML Model        │
    └──────────┬───────┘          └────────┬───────────────┘
               │                           │
               └──────────┬────────────────┘
                          ▼
              ┌───────────────────────┐
              │  Cleanup Intermediate │
              │       Files           │
              └───────────────────────┘
```

## Project Structure

```
.
├── .devcontainer/
│   └── devcontainer.json       # VS Code Dev Container configuration
├── dags/
│   └── etl_pipeline_dag.py     # Main Airflow DAG
├── data/                        # Data directory (created at runtime)
├── logs/                        # Airflow logs (created at runtime)
├── plugins/                     # Airflow plugins directory
├── scripts/                     # Utility scripts
├── docker-compose.yml          # Docker Compose configuration
├── Dockerfile                  # Custom Airflow image
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Prerequisites

- Docker and Docker Compose installed
- At least 4GB of available RAM
- Basic understanding of Apache Airflow concepts

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ids_706
```

### 2. Start Airflow

```bash
# Initialize Airflow (first time only)
docker-compose up airflow-init

# Start all services
docker-compose up -d
```

### 3. Access Airflow Web UI

- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

### 4. Verify Services

```bash
# Check running containers
docker-compose ps

# View logs
docker-compose logs -f airflow-scheduler
```

## DAG Details

### DAG Name: `etl_pipeline_dag`

- **Schedule**: Daily (`@daily`)
- **Start Date**: 2024-01-01
- **Catchup**: False
- **Tags**: etl, pipeline, parallel

### Tasks Overview

#### 1. Data Ingestion (Parallel)

- **`ingest_customers`**: Generates and ingests synthetic customer data (1,000 customers)
- **`ingest_orders`**: Generates and ingests synthetic order data (5,000 orders)

#### 2. Customer Transformation (TaskGroup)

- **`validate_customers`**: Removes duplicates, null values, and validates data quality
- **`enrich_customers`**: Adds age groups and customer segments

#### 3. Order Transformation (TaskGroup)

- **`validate_orders`**: Removes duplicates, null values, and validates data quality
- **`calculate_metrics`**: Calculates total amounts and order size categories

#### 4. Data Integration

- **`merge_data`**: Merges customer and order datasets on customer_id

#### 5. Database Loading

- **`load_to_postgres`**: Loads merged data into PostgreSQL table `customer_orders`

#### 6. Analysis (Parallel)

- **`perform_analysis`**:
  - Generates summary statistics by city, age group, and product
  - Calculates key metrics (revenue, average order value, etc.)
  - Saves results to JSON file

- **`train_ml_model`**:
  - Trains a Random Forest classifier to predict customer segments
  - Evaluates model performance
  - Calculates feature importance
  - Saves model results to JSON file

#### 7. Cleanup

- **`cleanup_intermediate_files`**: Removes intermediate CSV files, retains final outputs

### Data Flow

All tasks pass **file paths** (not actual data) through XCom to maintain efficiency and follow Airflow best practices.

## Key Datasets

### Customer Data

- `customer_id`: Unique identifier
- `name`: Customer name
- `email`: Customer email
- `age`: Customer age
- `city`: Customer location
- `registration_date`: Account registration date
- `age_group`: Derived age category
- `customer_segment`: Premium/Standard/Basic

### Order Data

- `order_id`: Unique identifier
- `customer_id`: Foreign key to customers
- `product`: Product name
- `quantity`: Order quantity
- `price`: Unit price
- `order_date`: Order timestamp
- `total_amount`: Calculated total (quantity × price)
- `order_size`: Small/Medium/Large/Extra Large

## Database Schema

### Table: `customer_orders`

The merged dataset contains all customer and order information with the following key columns:

- Customer fields: customer_id, name, email, age, city, age_group, customer_segment
- Order fields: order_id, product, quantity, price, total_amount, order_size
- Temporal fields: order_date, order_year, order_month

## Analysis Outputs

### 1. Analysis Results (`data/analysis_results.json`)

```json
{
  "total_revenue": 12345678.90,
  "total_orders": 5000,
  "average_order_value": 2469.14,
  "unique_customers": 995,
  "top_city": "New York",
  "top_product": "Product_A"
}
```

### 2. ML Model Results (`data/model_results.json`)

```json
{
  "accuracy": 0.34,
  "feature_importance": [
    {"feature": "total_amount", "importance": 0.45},
    {"feature": "quantity", "importance": 0.35},
    {"feature": "age", "importance": 0.20}
  ],
  "total_samples": 995,
  "train_samples": 796,
  "test_samples": 199
}
```

## Parallel Processing

The DAG is designed for maximum parallelism:

1. **Ingestion**: Customer and order data ingestion runs in parallel
2. **Transformation**: Customer and order transformations run in parallel TaskGroups
3. **Analysis**: Statistical analysis and ML training run in parallel

This parallel architecture significantly reduces total execution time.

## Best Practices Implemented

- **XCom for Paths Only**: Only file paths are passed between tasks, not actual data
- **Error Handling**: Retries configured with 5-minute delay
- **Data Validation**: Multiple validation steps ensure data quality
- **Resource Management**: Intermediate files cleaned up automatically
- **Modularity**: Each function has a single, well-defined responsibility
- **Documentation**: Comprehensive docstrings and comments

## Monitoring and Debugging

### View DAG in Airflow UI

1. Navigate to http://localhost:8080
2. Click on `etl_pipeline_dag`
3. View the Graph or Grid view to see task dependencies
4. Click on individual tasks to see logs and XCom values

### Check Logs

```bash
# Scheduler logs
docker-compose logs airflow-scheduler

# Webserver logs
docker-compose logs airflow-webserver

# Task logs
# Available in Airflow UI under each task
```

### Access PostgreSQL

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U airflow

# Query data
SELECT COUNT(*) FROM customer_orders;
SELECT city, COUNT(*) as customer_count FROM customer_orders GROUP BY city;
```

### Inspect Data Files

```bash
# List data files
docker-compose exec airflow-scheduler ls -lh /opt/airflow/data/

# View file contents
docker-compose exec airflow-scheduler cat /opt/airflow/data/analysis_results.json
```

## Troubleshooting

### Issue: Services won't start

```bash
# Stop all services
docker-compose down

# Remove volumes
docker-compose down -v

# Restart
docker-compose up airflow-init
docker-compose up -d
```

### Issue: Permission errors

```bash
# Check AIRFLOW_UID in .env matches your user
echo $UID

# Update .env file if needed
```

### Issue: DAG not appearing

```bash
# Check scheduler logs
docker-compose logs airflow-scheduler

# Verify DAG file syntax
docker-compose exec airflow-scheduler python /opt/airflow/dags/etl_pipeline_dag.py
```

## Stopping the Pipeline

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## Extension Ideas

### Optional Enhancements

1. **PySpark Integration** (15% Bonus):
   - Add PySpark for large-scale data transformations
   - Use Spark for parallel aggregations
   - Implement Spark ML for model training

2. **Additional Data Sources**:
   - Integrate with real APIs
   - Add support for CSV/JSON file uploads
   - Connect to external databases

3. **Advanced Analysis**:
   - Time series forecasting
   - Customer churn prediction
   - Product recommendation system

4. **Monitoring**:
   - Add Airflow alerts
   - Integrate with monitoring tools
   - Create custom metrics

## Technologies Used

- **Apache Airflow 2.7.3**: Workflow orchestration
- **PostgreSQL 13**: Data storage
- **Docker & Docker Compose**: Containerization
- **Python 3.11**: Programming language
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **SQLAlchemy**: Database ORM

## Performance Considerations

- Parallel task execution reduces total pipeline time by ~50%
- File-based XCom approach handles large datasets efficiently
- TaskGroups provide logical organization without overhead
- Cleanup task manages storage automatically

## License

This project is for educational purposes as part of IDS 706 coursework.

## Author

Created for IDS 706 - Data Engineering Systems

## Screenshots

### DAG Graph View

![DAG Graph](screenshots/dag_graph.png)

### Successful Execution

![DAG Success](screenshots/dag_success.png)

---

**Note**: Remember to trigger the DAG manually for the first run or wait for the scheduled execution!
