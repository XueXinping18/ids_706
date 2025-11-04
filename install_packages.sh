#!/bin/bash

# Install required Python packages in Airflow containers

echo "Installing Python packages in airflow-scheduler..."
docker-compose exec airflow-scheduler pip install pandas==2.1.3 numpy==1.26.2 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 psycopg2-binary==2.9.9 sqlalchemy==2.0.23

echo "Installing Python packages in airflow-webserver..."
docker-compose exec airflow-webserver pip install pandas==2.1.3 numpy==1.26.2 scikit-learn==1.3.2 matplotlib==3.8.2 seaborn==0.13.0 psycopg2-binary==2.9.9 sqlalchemy==2.0.23

echo "Packages installed successfully!"
echo "Restarting services to apply changes..."
docker-compose restart airflow-scheduler airflow-webserver

echo "Done! Airflow is ready."
