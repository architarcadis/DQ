import pytest
import pandas as pd
import numpy as np
from utils.suggest_keys import suggest_keys

def test_exact_match_columns():
    """Test identifying exact column name matches"""
    # Create test dataframes with an exact column match
    df1 = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        "value": [10, 20, 30, 40, 50]
    })
    
    df2 = pd.DataFrame({
        "id": [1, 2, 3, 6, 7],
        "category": ["A", "B", "C", "D", "E"],
        "score": [100, 200, 300, 400, 500]
    })
    
    # Get suggested keys
    suggestions = suggest_keys(df1, df2)
    
    # Check that the exact match is identified
    assert len(suggestions) > 0
    
    # Find the suggestion for 'id'
    id_suggestion = next((s for s in suggestions if s[0] == 'id' and s[1] == 'id'), None)
    
    # Verify it exists and has a high score
    assert id_suggestion is not None
    assert id_suggestion[2] > 90  # Should have a very high score

def test_fuzzy_match_columns():
    """Test identifying columns with similar names"""
    # Create test dataframes with similar column names
    df1 = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"]
    })
    
    df2 = pd.DataFrame({
        "customerid": [1, 2, 3, 6, 7],
        "product": ["A", "B", "C", "D", "E"]
    })
    
    # Get suggested keys
    suggestions = suggest_keys(df1, df2)
    
    # Check that the fuzzy match is identified
    assert len(suggestions) > 0
    
    # Find the suggestion for the similar columns
    fuzzy_suggestion = next((s for s in suggestions if s[0] == 'customer_id' and s[1] == 'customerid'), None)
    
    # Verify it exists and has a reasonable score
    assert fuzzy_suggestion is not None
    assert fuzzy_suggestion[2] > 70  # Should have a good score

def test_no_matching_columns():
    """Test behavior when no matches are found"""
    # Create test dataframes with no matching columns
    df1 = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "B", "C"]
    })
    
    df2 = pd.DataFrame({
        "completely_different": [4, 5, 6],
        "another_different": ["X", "Y", "Z"]
    })
    
    # Get suggested keys
    suggestions = suggest_keys(df1, df2)
    
    # Should return limited suggestions with low scores
    for suggestion in suggestions:
        assert suggestion[2] < 80  # Scores should be lower

def test_uniqueness_priority():
    """Test that columns with high uniqueness are prioritized"""
    # Create test dataframes with varying uniqueness
    df1 = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],  # Highly unique
        "category": ["A", "A", "B", "B", "C"],  # Less unique
        "constant": ["X", "X", "X", "X", "X"]  # Not unique at all
    })
    
    df2 = pd.DataFrame({
        "id": [1, 2, 3, 6, 7],
        "category": ["A", "B", "C", "D", "E"],
        "constant": ["Y", "Y", "Y", "Y", "Y"]
    })
    
    # Get suggested keys
    suggestions = suggest_keys(df1, df2)
    
    # The first suggestion should prioritize the id column
    assert suggestions[0][0] == 'id'
    
    # The constant column should not be in the top suggestions
    constant_suggestion = next((s for s in suggestions if s[0] == 'constant' and s[1] == 'constant'), None)
    
    # In this case, even though names match exactly, the uniqueness should be lower
    if constant_suggestion:
        assert constant_suggestion[2] < suggestions[0][2]

def test_value_overlap_impact():
    """Test that value overlap impacts the scores"""
    # Create test dataframes with different value overlaps
    df1 = pd.DataFrame({
        "id_high_overlap": [1, 2, 3, 4, 5],
        "id_low_overlap": [1, 2, 3, 4, 5]
    })
    
    df2 = pd.DataFrame({
        "id_high_overlap": [1, 2, 3, 6, 7],  # 60% overlap with df1
        "id_low_overlap": [8, 9, 10, 11, 12]  # 0% overlap with df1
    })
    
    # Get suggested keys
    suggestions = suggest_keys(df1, df2)
    
    # Find the two suggestions
    high_overlap = next((s for s in suggestions if s[0] == 'id_high_overlap'), None)
    low_overlap = next((s for s in suggestions if s[0] == 'id_low_overlap'), None)
    
    # The high overlap column should have a higher score
    assert high_overlap is not None
    assert low_overlap is not None
    assert high_overlap[2] > low_overlap[2]
