import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from gold_analysis import load_data, inspect_data, filter_and_group, ml_analysis, create_plots, main

class TestDataLoading:
    """Test data loading functionality"""
    
    def test_load_data_existing_file(self):
        """Test loading when file already exists"""
        with patch('os.path.exists', return_value=True):
            with patch('pandas.read_csv') as mock_csv:
                mock_csv.return_value = pd.DataFrame({'SPX': [1, 2, 3]})
                result = load_data()
                assert result is not None
                assert len(result) == 3
                mock_csv.assert_called_once()

    def test_load_data_kaggle_download(self):
        """Test loading with Kaggle download"""
        with patch('os.path.exists', return_value=False):
            with patch('kagglehub.dataset_download', return_value='/fake/path'):
                with patch('os.listdir', return_value=['test.csv']):
                    with patch('shutil.copy2'):
                        with patch('pandas.read_csv') as mock_csv:
                            mock_csv.return_value = pd.DataFrame({'SPX': [1, 2, 3]})
                            result = load_data()
                            assert result is not None

    def test_load_data_no_csv_files(self):
        """Edge case: No CSV files found after download"""
        with patch('os.path.exists', return_value=False):
            with patch('kagglehub.dataset_download', return_value='/fake/path'):
                with patch('os.listdir', return_value=[]):  # No CSV files
                    result = load_data()
                    assert result is None

class TestDataInspection:
    """Test data inspection and preprocessing"""
    
    def test_inspect_data_clean(self):
        """Test inspection with clean data"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        result = inspect_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_inspect_data_with_missing_values(self):
        """Edge case: Data with missing values"""
        df = pd.DataFrame({
            'A': [1, np.nan, 3],
            'B': [4, 5, np.nan]
        })
        result = inspect_data(df)
        assert isinstance(result, pd.DataFrame)
        assert result.isnull().sum().sum() == 2

    def test_inspect_data_with_duplicates(self):
        """Edge case: Data with duplicates"""
        df = pd.DataFrame({
            'A': [1, 1, 2],
            'B': [3, 3, 4]
        })
        result = inspect_data(df)
        assert isinstance(result, pd.DataFrame)
        assert result.duplicated().sum() == 1

class TestFiltering:
    """Test filtering and grouping functionality"""
    
    def test_filter_and_group_with_dates(self):
        """Test filtering with proper date column"""
        df = pd.DataFrame({
            'SPX': [100, 200, 300, 400],
            'Date': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
        })
        result = filter_and_group(df)
        assert isinstance(result, pd.DataFrame)
        assert 'year' in result.columns

    def test_filter_and_group_no_numeric_columns(self):
        """Edge case: No numeric columns"""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'category': ['x', 'y', 'z']
        })
        result = filter_and_group(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_filter_and_group_invalid_dates(self):
        """Edge case: Invalid date column"""
        df = pd.DataFrame({
            'SPX': [100, 200, 300],
            'Date': ['invalid', 'date', 'values']
        })
        result = filter_and_group(df)
        assert isinstance(result, pd.DataFrame)
        # Should not have year column due to invalid dates
        assert 'year' not in result.columns or result['year'].isnull().all()

class TestMachineLearning:
    """Test machine learning functionality"""
    
    def test_ml_analysis_sufficient_data(self):
        """Test ML with sufficient data"""
        df = pd.DataFrame({
            'SPX': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'GLD': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        models = ml_analysis(df)
        assert len(models) == 6
        assert models[0] is not None  # lr model
        assert models[1] is not None  # rf model
        assert models[2] is not None  # X_test
        assert models[3] is not None  # y_test

    def test_ml_analysis_insufficient_data(self):
        """Edge case: Insufficient numeric data"""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c']
        })
        models = ml_analysis(df)
        assert all(x is None for x in models)

    def test_ml_analysis_single_column(self):
        """Edge case: Only one numeric column"""
        df = pd.DataFrame({
            'SPX': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        })
        models = ml_analysis(df)
        # Should create time_index feature and still work
        assert models[0] is not None
        assert models[1] is not None

class TestVisualization:
    """Test plotting functionality"""
    
    def test_create_plots_with_data(self):
        """Test plotting with proper data"""
        df = pd.DataFrame({
            'SPX': [100, 200, 300],
            'year': [2020, 2021, 2022]
        })
        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.savefig') as mock_save:
                create_plots(df)
                mock_save.assert_called_once_with('gold_analysis.png')

    def test_create_plots_no_numeric_data(self):
        """Edge case: No numeric data for plotting"""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c']
        })
        with patch('matplotlib.pyplot.show'):
            # Should handle gracefully without errors
            create_plots(df)

class TestSystemIntegration:
    """System tests for complete pipeline"""
    
    def test_full_analysis_pipeline(self):
        """System test: Complete analysis pipeline"""
        sample_df = pd.DataFrame({
            'Date': ['2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', 
                    '2020-06-01', '2021-06-01', '2022-06-01', '2023-06-01'],
            'SPX': [3000, 3200, 3100, 3400, 3050, 3250, 3150, 3450],
            'GLD': [150, 160, 155, 165, 152, 162, 157, 167]
        })
        
        with patch('gold_analysis.load_data', return_value=sample_df):
            with patch('matplotlib.pyplot.show'):
                with patch('matplotlib.pyplot.savefig'):
                    # Should run entire pipeline without errors
                    main()

    def test_pipeline_with_data_loading_failure(self):
        """System test: Handle data loading failure"""
        with patch('gold_analysis.load_data', return_value=None):
            # Should handle gracefully when data loading fails
            main()
