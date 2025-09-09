# Gold Price Data Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import kagglehub
import shutil

def load_data():
    local_file = "gold_data_2015_25.csv"
    
    if not os.path.exists(local_file):
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mdanwarhossain200110/gold-price-2015-2025")
        print(f"Downloaded to: {path}")
        
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if csv_files:
            source_file = os.path.join(path, csv_files[0])
            shutil.copy2(source_file, local_file)
            print(f"Copied {csv_files[0]} to {local_file}")
        else:
            print("No CSV files found")
            return None
    else:
        print("Using existing dataset file")
    
    df = pd.read_csv(local_file)
    print("Data loaded successfully")
    print(df.head())
    return df

def inspect_data(df):
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Stats:")
    print(df.describe())
    print("\nMissing values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())
    return df

def filter_and_group(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found")
        return df
    
    price_col = numeric_cols[0]
    for col in numeric_cols:
        if 'price' in col.lower() or 'close' in col.lower():
            price_col = col
            break
    
    print(f"\nUsing {price_col} as main metric")
    
    # High value filter
    threshold = df[price_col].quantile(0.75)
    high_data = df[df[price_col] > threshold]
    print(f"High value periods (>{threshold:.2f}): {len(high_data)} records")
    
    # Try to create year column
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        try:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df['year'] = df[date_cols[0]].dt.year
            yearly_stats = df.groupby('year')[price_col].agg(['mean', 'count'])
            print(f"\nYearly stats:")
            print(yearly_stats)
        except:
            print("Could not process date column")
    
    return df

def ml_analysis(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        print("Not enough numeric data for ML")
        return None, None, None, None, None, None
    
    target_col = numeric_cols[0]
    for col in numeric_cols:
        if 'price' in col.lower() or 'close' in col.lower():
            target_col = col
            break
    
    feature_cols = [col for col in numeric_cols if col != target_col]
    if not feature_cols:
        df['time_index'] = range(len(df))
        feature_cols = ['time_index']
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col].fillna(df[target_col].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    
    print(f"\nML Results:")
    print(f"Linear Regression R²: {lr_r2:.4f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    
    return lr, rf, X_test, y_test, lr_pred, rf_pred

def create_plots(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns for plotting")
        return
    
    price_col = numeric_cols[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Histogram
    axes[0, 0].hist(df[price_col], bins=30, alpha=0.7)
    axes[0, 0].set_title(f'{price_col} Distribution')
    axes[0, 0].set_xlabel(price_col)
    axes[0, 0].set_ylabel('Frequency')
    
    # Box plot
    axes[0, 1].boxplot(df[price_col])
    axes[0, 1].set_title(f'{price_col} Box Plot')
    axes[0, 1].set_ylabel(price_col)
    
    # Time series
    if 'year' in df.columns:
        yearly_avg = df.groupby('year')[price_col].mean()
        axes[1, 0].plot(yearly_avg.index, yearly_avg.values, marker='o')
        axes[1, 0].set_title(f'Average {price_col} by Year')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel(f'Average {price_col}')
    else:
        axes[1, 0].plot(df[price_col])
        axes[1, 0].set_title(f'{price_col} Over Time')
    
    # Scatter
    axes[1, 1].scatter(range(len(df)), df[price_col], alpha=0.6)
    axes[1, 1].set_title(f'{price_col} Scatter')
    axes[1, 1].set_xlabel('Index')
    axes[1, 1].set_ylabel(price_col)
    
    plt.tight_layout()
    plt.savefig('gold_analysis.png')
    plt.show()

def main():
    print("Starting Gold Price Analysis...")
    
    df = load_data()
    if df is None:
        print("Failed to load data")
        return
    
    df = inspect_data(df)
    df = filter_and_group(df)
    
    ml_results = ml_analysis(df)
    create_plots(df)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
