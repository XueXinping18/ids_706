# Gold Price Data Analysis

This is my data analysis project for class. I'm analyzing gold price data from 2015-2025 and using Docker to make it reproducible.

## Dataset
Source: https://www.kaggle.com/datasets/mdanwarhossain200110/gold-price-2015-2025
File: `gold_data_2015_25.csv` (auto-downloaded from Kaggle)

## Setup

### Docker Setup 
```bash
# Build the container
make docker-build

# Run the analysis
make docker-run
```

### Local Setup (not removed for reference)
```bash
make install
make analyze
```

## Testing

I wrote comprehensive tests covering all the core functions with edge case handling. Run with:
```bash
make test
```

**Test Coverage:**
- **Data Loading Tests** (3 tests) - File exists locally, Kaggle download, and no CSV files found
- **Data Inspection Tests** (3 tests) - Clean data, missing values, and duplicate records  
- **Filtering Tests** (3 tests) - Proper dates, no numeric columns, and invalid date formats
- **Machine Learning Tests** (3 tests) - Sufficient data, insufficient data, and single column cases
- **Visualization Tests** (2 tests) - Normal plotting and no numeric data edge case
- **System Tests** (2 tests) - Complete pipeline and data loading failure handling

The tests are organized into classes by functionality and include both positive cases (normal operation) and negative cases (edge cases and error conditions). All tests use mock data so they run quickly without needing the actual Kaggle dataset or external dependencies.

## Files
- `gold_analysis.py` - main analysis script
- `test_gold_analysis.py` - test suite for the functions
- `Dockerfile` - Docker container setup
- `requirements.txt` - Python dependencies
- `Makefile` - build automation
- `gold_data_2015_25.csv` - dataset (auto-downloaded from Kaggle)

## Available Make Commands
```bash
make install      # Install dependencies
make format       # Format code with black
make lint         # Lint code with flake8
make test         # Run tests
make analyze      # Run data analysis
make docker-build # Build Docker image
make docker-run   # Run in Docker
make clean        # Clean generated files
make all          # Run install, format, lint, test
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

## Docker Setup

Simple Docker container with Python 3.12 and all the packages needed.

Build and run:
```bash
docker build -t gold-analysis .
docker run --rm gold-analysis
```

The Dockerfile installs all requirements and runs the analysis automatically.

## Generated Files
- `gold_analysis.png` - visualization dashboard showing SPX distribution, trends, and patterns

![Gold Analysis Results](gold_analysis.png)
![Test results](test_results.png)
