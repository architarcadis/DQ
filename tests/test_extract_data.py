import pytest
import pandas as pd
import io
from utils.extract_data import extract_data

# Mock file objects for testing
class MockFile:
    def __init__(self, name, content, content_type="text/plain"):
        self.name = name
        self.content = content
        self.content_type = content_type
    
    def read(self):
        return self.content
    
    def getvalue(self):
        return self.content

def test_extract_csv():
    """Test CSV extraction functionality"""
    # Create a mock CSV file
    csv_content = b"col1,col2,col3\n1,2,3\n4,5,6\n7,8,9"
    mock_file = MockFile("test.csv", csv_content)
    
    # Monkeypatch pd.read_csv to use StringIO
    original_read_csv = pd.read_csv
    pd.read_csv = lambda f: original_read_csv(io.BytesIO(f.content))
    
    try:
        # Test extraction
        result = extract_data([mock_file])
        
        # Check results
        assert "test" in result
        assert isinstance(result["test"], pd.DataFrame)
        assert result["test"].shape == (3, 3)
        assert list(result["test"].columns) == ["col1", "col2", "col3"]
    finally:
        # Restore original function
        pd.read_csv = original_read_csv

def test_extract_excel():
    """Test Excel extraction functionality"""
    # This is a simplified test as we can't easily create Excel binary content
    # In a real test, you would use a fixture with a real Excel file
    
    # Mocked expected output
    expected_df = pd.DataFrame({
        "col1": [1, 4, 7],
        "col2": [2, 5, 8],
        "col3": [3, 6, 9]
    })
    
    # Mock Excel reading function to return our test DataFrame
    original_excel_file = pd.ExcelFile
    pd.ExcelFile = lambda f: type('obj', (object,), {
        'sheet_names': ['Sheet1'],
        'parse': lambda sheet_name, **kwargs: expected_df
    })
    
    # Mock read_excel
    original_read_excel = pd.read_excel
    pd.read_excel = lambda *args, **kwargs: expected_df
    
    try:
        # Create a mock Excel file
        mock_file = MockFile("test.xlsx", b"dummy content")
        
        # Test extraction
        result = extract_data([mock_file])
        
        # Check results
        assert "test" in result
        assert isinstance(result["test"], pd.DataFrame)
        assert result["test"].shape == (3, 3)
    finally:
        # Restore original functions
        pd.ExcelFile = original_excel_file
        pd.read_excel = original_read_excel

def test_multiple_files():
    """Test handling multiple files at once"""
    # Create mock files
    csv1_content = b"col1,col2\n1,2\n3,4"
    csv2_content = b"colA,colB\n10,20\n30,40"
    
    mock_file1 = MockFile("file1.csv", csv1_content)
    mock_file2 = MockFile("file2.csv", csv2_content)
    
    # Monkeypatch pd.read_csv
    original_read_csv = pd.read_csv
    
    def mock_read_csv(f):
        if f.name == "file1.csv":
            return original_read_csv(io.BytesIO(csv1_content))
        else:
            return original_read_csv(io.BytesIO(csv2_content))
    
    pd.read_csv = mock_read_csv
    
    try:
        # Test extraction of multiple files
        result = extract_data([mock_file1, mock_file2])
        
        # Check results
        assert len(result) == 2
        assert "file1" in result
        assert "file2" in result
        assert result["file1"].shape == (2, 2)
        assert result["file2"].shape == (2, 2)
    finally:
        # Restore original function
        pd.read_csv = original_read_csv

def test_error_handling():
    """Test error handling for invalid files"""
    # Create an invalid CSV file
    invalid_content = b"This is not a valid CSV file"
    mock_file = MockFile("invalid.csv", invalid_content)
    
    # Monkeypatch pd.read_csv to raise an exception
    def mock_read_csv_error(f):
        raise pd.errors.ParserError("Error parsing CSV")
    
    original_read_csv = pd.read_csv
    pd.read_csv = mock_read_csv_error
    
    try:
        # Test extraction with error
        result = extract_data([mock_file])
        
        # Should return empty dict as the file is invalid
        assert result == {}
    finally:
        # Restore original function
        pd.read_csv = original_read_csv
