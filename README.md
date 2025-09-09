# Gold Price Data Analysis

## Dataset
Source: https://www.kaggle.com/datasets/mdanwarhossain200110/gold-price-2015-2025
File: `gold_data_2015_25.csv`

## Setup
```bash
make install
```

## Files
- `gold_analysis.py` - main analysis script
- `gold_data_2015_25.csv` - dataset (auto-downloaded from Kaggle)
- `README.md` - this file
- `Makefile` - build automation

## Run Analysis
```bash
make analyze
```

## Available Make Commands
```bash
make install    # Install dependencies
make format     # Format code with black
make lint       # Lint code with flake8
make test       # Run tests
make analyze    # Run data analysis
make clean      # Clean generated files
make all        # Run install, format, lint, test
```

## What it does

### 1. Import Dataset
Loads the gold price CSV file and displays first few rows

### 2. Data Inspection
- Shows dataset info with `.info()` and `.describe()`
- Checks for missing values and duplicates
- Basic data quality assessment

### 3. Filtering and Grouping
- Filters for high-price periods (top 25%)
- Groups data by year for trend analysis
- Calculates yearly statistics (mean, std, count)

### 4. Machine Learning
- Tests Linear Regression and Random Forest models
- Splits data 80/20 for training/testing
- Reports R² scores for both models
- Uses available numeric features to predict prices

### 5. Visualization
Creates 4 plots:
- Histogram of price distribution
- Box plot for outlier detection
- Time series of yearly averages
- Scatter plot over time

## Key Findings

### Dataset Overview
- **Total Records**: 2,666 entries (2015-2025)
- **Data Quality**: No missing values, no duplicates
- **Main Metrics**: SPX, GLD (gold), USO (oil), SLV (silver), EUR/USD

### Market Analysis
- **SPX Range**: 1,829 to 6,468 points
- **High Performance Periods**: 667 records above 4,373 points (top 25%)
- **Best Year**: 2025 with average SPX of 5,922 points
- **Growth Trend**: Consistent upward trend from 2015-2025

### Yearly Performance
| Year | Average SPX | Records |
|------|-------------|---------|
| 2015 | 2,061 | 252 |
| 2016 | 2,095 | 252 |
| 2017 | 2,449 | 249 |
| 2018 | 2,746 | 251 |
| 2019 | 2,914 | 251 |
| 2020 | 3,218 | 253 |
| 2021 | 4,273 | 252 |
| 2022 | 4,099 | 251 |
| 2023 | 4,284 | 250 |
| 2024 | 5,428 | 252 |
| 2025 | 5,922 | 153 |

### Machine Learning Results
- **Linear Regression R²**: 0.9574 (95.74% accuracy)
- **Random Forest R²**: 0.9959 (99.59% accuracy)
- **Best Model**: Random Forest shows superior predictive performance
- **Model Performance**: Both models show excellent predictive capability

## Generated Files
- `gold_analysis.png` - visualization dashboard showing SPX distribution, trends, and patterns
