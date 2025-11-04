"""
ETL Pipeline DAG for Customer and Order Data Processing
This DAG demonstrates parallel processing, data transformation, merging, and analysis.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import json

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'etl_pipeline_dag',
    default_args=default_args,
    description='ETL pipeline with parallel processing for customer and order data',
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'pipeline', 'parallel'],
)

# Data directory
DATA_DIR = '/opt/airflow/data'

# PostgreSQL connection
DB_CONN = 'postgresql://airflow:airflow@postgres:5432/airflow'


def ingest_customer_data(**context):
    """Ingest customer data - simulates data ingestion from a source"""
    print("Ingesting customer data...")

    # Generate synthetic customer data
    np.random.seed(42)
    n_customers = 1000

    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'name': [f'Customer_{i}' for i in range(1, n_customers + 1)],
        'email': [f'customer_{i}@email.com' for i in range(1, n_customers + 1)],
        'age': np.random.randint(18, 80, n_customers),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers),
        'registration_date': pd.date_range(start='2020-01-01', periods=n_customers, freq='D')
    })

    # Save to CSV
    file_path = f'{DATA_DIR}/customers_raw.csv'
    customers.to_csv(file_path, index=False)
    print(f"Customer data saved to {file_path}")
    print(f"Total customers: {len(customers)}")

    # Push file path to XCom (not the data itself)
    return file_path


def ingest_order_data(**context):
    """Ingest order data - simulates data ingestion from a source"""
    print("Ingesting order data...")

    # Generate synthetic order data
    np.random.seed(42)
    n_orders = 5000

    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'customer_id': np.random.randint(1, 1001, n_orders),
        'product': np.random.choice(['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E'], n_orders),
        'quantity': np.random.randint(1, 10, n_orders),
        'price': np.random.uniform(10, 500, n_orders).round(2),
        'order_date': pd.date_range(start='2020-01-01', periods=n_orders, freq='H')
    })

    # Save to CSV
    file_path = f'{DATA_DIR}/orders_raw.csv'
    orders.to_csv(file_path, index=False)
    print(f"Order data saved to {file_path}")
    print(f"Total orders: {len(orders)}")

    # Push file path to XCom
    return file_path


def validate_customer_data(**context):
    """Validate customer data"""
    file_path = context['ti'].xcom_pull(task_ids='ingest_customers')
    print(f"Validating customer data from {file_path}")

    df = pd.read_csv(file_path)

    # Remove duplicates
    df = df.drop_duplicates(subset=['customer_id'])

    # Remove null values
    df = df.dropna()

    # Validate age range
    df = df[(df['age'] >= 18) & (df['age'] <= 100)]

    validated_path = f'{DATA_DIR}/customers_validated.csv'
    df.to_csv(validated_path, index=False)
    print(f"Validated {len(df)} customers")

    return validated_path


def enrich_customer_data(**context):
    """Enrich customer data with additional features"""
    file_path = context['ti'].xcom_pull(task_ids='customer_transformation.validate_customers')
    print(f"Enriching customer data from {file_path}")

    df = pd.read_csv(file_path)

    # Add age group
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100],
                              labels=['18-25', '26-35', '36-50', '50+'])

    # Add customer segment
    df['customer_segment'] = np.random.choice(['Premium', 'Standard', 'Basic'], len(df))

    enriched_path = f'{DATA_DIR}/customers_enriched.csv'
    df.to_csv(enriched_path, index=False)
    print(f"Enriched {len(df)} customers")

    return enriched_path


def validate_order_data(**context):
    """Validate order data"""
    file_path = context['ti'].xcom_pull(task_ids='ingest_orders')
    print(f"Validating order data from {file_path}")

    df = pd.read_csv(file_path)

    # Remove duplicates
    df = df.drop_duplicates(subset=['order_id'])

    # Remove null values
    df = df.dropna()

    # Validate positive values
    df = df[(df['quantity'] > 0) & (df['price'] > 0)]

    validated_path = f'{DATA_DIR}/orders_validated.csv'
    df.to_csv(validated_path, index=False)
    print(f"Validated {len(df)} orders")

    return validated_path


def calculate_order_metrics(**context):
    """Calculate metrics for each order"""
    file_path = context['ti'].xcom_pull(task_ids='order_transformation.validate_orders')
    print(f"Calculating order metrics from {file_path}")

    df = pd.read_csv(file_path)

    # Calculate total amount
    df['total_amount'] = df['quantity'] * df['price']

    # Add order size category
    df['order_size'] = pd.cut(df['total_amount'], bins=[0, 100, 500, 1000, np.inf],
                               labels=['Small', 'Medium', 'Large', 'Extra Large'])

    metrics_path = f'{DATA_DIR}/orders_with_metrics.csv'
    df.to_csv(metrics_path, index=False)
    print(f"Calculated metrics for {len(df)} orders")

    return metrics_path


def merge_datasets(**context):
    """Merge customer and order datasets"""
    customer_path = context['ti'].xcom_pull(task_ids='customer_transformation.enrich_customers')
    order_path = context['ti'].xcom_pull(task_ids='order_transformation.calculate_metrics')

    print(f"Merging datasets from {customer_path} and {order_path}")

    customers = pd.read_csv(customer_path)
    orders = pd.read_csv(order_path)

    # Merge datasets
    merged = pd.merge(orders, customers, on='customer_id', how='left')

    # Add derived features
    merged['order_year'] = pd.to_datetime(merged['order_date']).dt.year
    merged['order_month'] = pd.to_datetime(merged['order_date']).dt.month

    merged_path = f'{DATA_DIR}/merged_data.csv'
    merged.to_csv(merged_path, index=False)
    print(f"Merged dataset created with {len(merged)} records")

    return merged_path


def load_to_postgres(**context):
    """Load merged data to PostgreSQL"""
    file_path = context['ti'].xcom_pull(task_ids='merge_data')
    print(f"Loading data to PostgreSQL from {file_path}")

    df = pd.read_csv(file_path)

    # Create database engine
    engine = create_engine(DB_CONN)

    # Load data to PostgreSQL
    table_name = 'customer_orders'
    df.to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"Loaded {len(df)} records to table '{table_name}'")

    return table_name


def perform_analysis(**context):
    """Perform data analysis and create visualizations"""
    print("Performing data analysis...")

    # Read data from PostgreSQL
    engine = create_engine(DB_CONN)
    query = "SELECT * FROM customer_orders"
    df = pd.read_sql(query, engine)

    print(f"Loaded {len(df)} records from database")

    # Analysis 1: Summary statistics by city
    city_summary = df.groupby('city').agg({
        'total_amount': ['sum', 'mean', 'count']
    }).round(2)

    print("\nSummary by City:")
    print(city_summary)

    # Analysis 2: Summary by age group
    age_group_summary = df.groupby('age_group').agg({
        'total_amount': ['sum', 'mean', 'count']
    }).round(2)

    print("\nSummary by Age Group:")
    print(age_group_summary)

    # Analysis 3: Summary by product
    product_summary = df.groupby('product').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum'
    }).round(2)

    print("\nSummary by Product:")
    print(product_summary)

    # Save analysis results
    analysis_results = {
        'total_revenue': float(df['total_amount'].sum()),
        'total_orders': int(len(df)),
        'average_order_value': float(df['total_amount'].mean()),
        'unique_customers': int(df['customer_id'].nunique()),
        'top_city': df.groupby('city')['total_amount'].sum().idxmax(),
        'top_product': df.groupby('product')['total_amount'].sum().idxmax()
    }

    results_path = f'{DATA_DIR}/analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\nAnalysis results saved to {results_path}")
    print(f"Analysis Summary: {analysis_results}")

    return results_path


def train_ml_model(**context):
    """Train a simple machine learning model"""
    print("Training machine learning model...")

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    # Read data from PostgreSQL
    engine = create_engine(DB_CONN)
    query = "SELECT * FROM customer_orders"
    df = pd.read_sql(query, engine)

    # Prepare features for predicting customer segment
    # Select only customers with complete data
    df_model = df[['age', 'total_amount', 'quantity', 'customer_segment']].dropna()

    # Add aggregated features per customer
    customer_features = df.groupby('customer_id').agg({
        'age': 'first',
        'total_amount': 'sum',
        'quantity': 'sum',
        'customer_segment': 'first'
    }).reset_index()

    # Prepare features and target
    X = customer_features[['age', 'total_amount', 'quantity']]
    y = customer_features['customer_segment']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Save model results
    model_results = {
        'accuracy': float(accuracy),
        'feature_importance': feature_importance.to_dict('records'),
        'total_samples': len(customer_features),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    results_path = f'{DATA_DIR}/model_results.json'
    with open(results_path, 'w') as f:
        json.dump(model_results, f, indent=2)

    print(f"\nModel results saved to {results_path}")

    return results_path


def cleanup_intermediate_files(**context):
    """Clean up intermediate files"""
    print("Cleaning up intermediate files...")

    # List of intermediate files to remove
    intermediate_files = [
        f'{DATA_DIR}/customers_raw.csv',
        f'{DATA_DIR}/orders_raw.csv',
        f'{DATA_DIR}/customers_validated.csv',
        f'{DATA_DIR}/orders_validated.csv',
        f'{DATA_DIR}/customers_enriched.csv',
        f'{DATA_DIR}/orders_with_metrics.csv',
    ]

    removed_count = 0
    for file_path in intermediate_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed: {file_path}")
            removed_count += 1

    print(f"\nCleaned up {removed_count} intermediate files")

    # Keep final files
    final_files = [
        f'{DATA_DIR}/merged_data.csv',
        f'{DATA_DIR}/analysis_results.json',
        f'{DATA_DIR}/model_results.json'
    ]

    print(f"\nFinal files retained:")
    for file_path in final_files:
        if os.path.exists(file_path):
            print(f"  - {file_path}")

    return removed_count


# Task 1: Ingest customer data
ingest_customers_task = PythonOperator(
    task_id='ingest_customers',
    python_callable=ingest_customer_data,
    dag=dag,
)

# Task 2: Ingest order data (parallel with Task 1)
ingest_orders_task = PythonOperator(
    task_id='ingest_orders',
    python_callable=ingest_order_data,
    dag=dag,
)

# TaskGroup for customer data transformation
with TaskGroup('customer_transformation', dag=dag) as customer_transformation:
    validate_customers_task = PythonOperator(
        task_id='validate_customers',
        python_callable=validate_customer_data,
    )

    enrich_customers_task = PythonOperator(
        task_id='enrich_customers',
        python_callable=enrich_customer_data,
    )

    validate_customers_task >> enrich_customers_task

# TaskGroup for order data transformation
with TaskGroup('order_transformation', dag=dag) as order_transformation:
    validate_orders_task = PythonOperator(
        task_id='validate_orders',
        python_callable=validate_order_data,
    )

    calculate_metrics_task = PythonOperator(
        task_id='calculate_metrics',
        python_callable=calculate_order_metrics,
    )

    validate_orders_task >> calculate_metrics_task

# Task: Merge datasets
merge_data_task = PythonOperator(
    task_id='merge_data',
    python_callable=merge_datasets,
    dag=dag,
)

# Task: Load to PostgreSQL
load_postgres_task = PythonOperator(
    task_id='load_to_postgres',
    python_callable=load_to_postgres,
    dag=dag,
)

# Task: Perform analysis (parallel with ML training)
analysis_task = PythonOperator(
    task_id='perform_analysis',
    python_callable=perform_analysis,
    dag=dag,
)

# Task: Train ML model (parallel with analysis)
ml_task = PythonOperator(
    task_id='train_ml_model',
    python_callable=train_ml_model,
    dag=dag,
)

# Task: Cleanup intermediate files
cleanup_task = PythonOperator(
    task_id='cleanup_intermediate_files',
    python_callable=cleanup_intermediate_files,
    dag=dag,
)

# Define task dependencies
# Parallel ingestion
[ingest_customers_task, ingest_orders_task]

# Parallel transformation groups
ingest_customers_task >> customer_transformation
ingest_orders_task >> order_transformation

# Merge after both transformations complete
[customer_transformation, order_transformation] >> merge_data_task

# Load to database
merge_data_task >> load_postgres_task

# Parallel analysis and ML training
load_postgres_task >> [analysis_task, ml_task]

# Cleanup after all analysis is done
[analysis_task, ml_task] >> cleanup_task
