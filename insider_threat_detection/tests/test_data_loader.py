"""Tests for data loader module."""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_loader = DataLoader(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_csv(self, filename: str, data: dict):
        """Create a sample CSV file for testing."""
        df = pd.DataFrame(data)
        filepath = os.path.join(self.temp_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def test_load_single_file_success(self):
        """Test successful loading of a single file."""
        # Create sample data
        sample_data = {
            'id': [1, 2, 3],
            'date': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00'],
            'user': ['user1', 'user2', 'user1'],
            'pc': ['pc1', 'pc2', 'pc1'],
            'url': ['http://example.com', 'http://test.com', 'http://sample.com']
        }
        
        self.create_sample_csv('http.csv', sample_data)
        
        # Test loading
        df = self.data_loader.load_single_file('http', ['http.csv'])
        
        assert df is not None
        assert len(df) == 3
        assert 'activity_type' in df.columns
        assert df['activity_type'].iloc[0] == 'HTTP'
        assert 'details' in df.columns  # url should be renamed to details
    
    def test_load_single_file_not_found(self):
        """Test loading when file doesn't exist."""
        df = self.data_loader.load_single_file('http', ['nonexistent.csv'])
        assert df is None
    
    def test_standardize_dataframe(self):
        """Test dataframe standardization."""
        # Create dataframe with missing columns
        df = pd.DataFrame({
            'id': [1, 2],
            'date': ['2023-01-01', '2023-01-02'],
            'user': ['user1', 'user2']
        })
        
        standardized_df = self.data_loader.standardize_dataframe(df)
        
        # Check all required columns are present
        required_columns = ['id', 'date', 'user', 'pc', 'activity_type', 'details']
        for col in required_columns:
            assert col in standardized_df.columns
        
        # Check missing columns are filled with 'unknown'
        assert standardized_df['pc'].iloc[0] == 'unknown'
        assert standardized_df['activity_type'].iloc[0] == 'unknown'
        assert standardized_df['details'].iloc[0] == 'unknown'
    
    def test_load_and_merge_data_success(self):
        """Test successful data loading and merging."""
        # Create sample HTTP data
        http_data = {
            'id': [1, 2],
            'date': ['2023-01-01 10:00:00', '2023-01-01 11:00:00'],
            'user': ['user1', 'user2'],
            'pc': ['pc1', 'pc2'],
            'url': ['http://example.com', 'http://test.com']
        }
        self.create_sample_csv('http.csv', http_data)
        
        # Create sample device data
        device_data = {
            'id': [3, 4],
            'date': ['2023-01-01 12:00:00', '2023-01-01 13:00:00'],
            'user': ['user1', 'user3'],
            'pc': ['pc1', 'pc3'],
            'activity': ['file_access', 'usb_insert']
        }
        self.create_sample_csv('device.csv', device_data)
        
        # Test loading and merging
        merged_df = self.data_loader.load_and_merge_data()
        
        assert merged_df is not None
        assert len(merged_df) == 4  # 2 HTTP + 2 device records
        assert 'HTTP' in merged_df['activity_type'].values
        assert 'DEVICE' in merged_df['activity_type'].values
    
    def test_load_and_merge_data_no_files(self):
        """Test loading when no data files exist."""
        with pytest.raises(ValueError, match="No data files found"):
            self.data_loader.load_and_merge_data()
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        # Create and load sample data
        sample_data = {
            'id': [1, 2, 3],
            'date': ['2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00'],
            'user': ['user1', 'user2', 'user1'],
            'pc': ['pc1', 'pc2', 'pc1'],
            'url': ['http://example.com', 'http://test.com', 'http://sample.com']
        }
        self.create_sample_csv('http.csv', sample_data)
        
        merged_df = self.data_loader.load_and_merge_data()
        summary = self.data_loader.get_data_summary()
        
        assert summary['total_records'] == 3
        assert summary['unique_users'] == 2
        assert summary['unique_pcs'] == 2
        assert 'HTTP' in summary['activity_types']
        assert summary['activity_types']['HTTP'] == 3
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Create sample data with some quality issues
        sample_data = {
            'id': [1, 2, 3],
            'date': ['2023-01-01 10:00:00', 'unknown', '2023-01-01 12:00:00'],
            'user': ['user1', 'user2', None],
            'pc': ['pc1', 'pc2', 'pc1'],
            'url': ['http://example.com', 'http://test.com', 'http://sample.com']
        }
        self.create_sample_csv('http.csv', sample_data)
        
        merged_df = self.data_loader.load_and_merge_data()
        validation_results = self.data_loader.validate_data_quality()
        
        assert 'issues' in validation_results
        assert validation_results['total_records'] == 3
        
        # Should detect unknown dates
        issues = validation_results['issues']
        unknown_date_issue = any('Unknown dates found' in issue for issue in issues)
        assert unknown_date_issue


if __name__ == "__main__":
    pytest.main([__file__])
