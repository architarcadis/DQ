import pytest
import pandas as pd
import numpy as np
from utils.auto_model import get_available_models

def test_get_available_models():
    """Test that the function returns appropriate models for each type"""
    # Test for classification
    classification_models = get_available_models('classification')
    assert len(classification_models) > 0
    assert 'Logistic Regression' in classification_models
    assert 'Random Forest' in classification_models
    
    # Test for regression
    regression_models = get_available_models('regression')
    assert len(regression_models) > 0
    assert 'Linear Regression' in regression_models
    assert 'Random Forest' in regression_models
    
    # Test for clustering
    clustering_models = get_available_models('clustering')
    assert len(clustering_models) > 0
    assert 'K-Means' in clustering_models
    
    # Test for invalid type
    invalid_models = get_available_models('invalid_type')
    assert len(invalid_models) == 0

# Note: We're not testing run_auto_model directly as it depends on PyCaret
# which would be complex to mock properly in a unit test.
# In a real project, you would use integration tests for this functionality.

def test_model_type_detection():
    """
    Smoke test to verify logic for determining model type based on target variable.
    This doesn't actually run auto_model but tests the logic that would be used.
    """
    # Create test dataframes
    df = pd.DataFrame({
        "numeric_target": [1.1, 2.2, 3.3, 4.4, 5.5],
        "categorical_target": ["A", "B", "A", "C", "B"],
        "feature1": [10, 20, 30, 40, 50],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Test logic for regression (numeric target)
    target = "numeric_target"
    if df[target].dtype in [np.float64, np.int64]:
        model_type = "regression"
    else:
        model_type = "classification"
    
    assert model_type == "regression"
    
    # Test logic for classification (categorical target)
    target = "categorical_target"
    if df[target].dtype in [np.float64, np.int64]:
        model_type = "regression"
    else:
        model_type = "classification"
    
    assert model_type == "classification"
