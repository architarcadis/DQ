import pandas as pd
import numpy as np
import streamlit as st

# Try to import fuzzywuzzy, but gracefully handle if it's not available
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

@st.cache_data
def suggest_keys(df1, df2, threshold=70):
    """
    Identify potential primary key / foreign key relationships between two dataframes.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First dataframe (considered as primary)
    df2 : pandas.DataFrame
        Second dataframe (considered as secondary)
    threshold : int, optional
        Minimum score threshold for fuzzy matching column names (default: 70)
    
    Returns:
    --------
    list of tuples
        Each tuple contains (primary_key, foreign_key, score) representing potential join keys
    """
    suggested_keys = []
    
    # Helper function to calculate uniqueness score (0-1)
    def uniqueness_score(series):
        if len(series) == 0:
            return 0
        return series.nunique() / len(series)
    
    # Calculate uniqueness scores for all columns in both dataframes
    df1_uniqueness = {col: uniqueness_score(df1[col]) for col in df1.columns}
    df2_uniqueness = {col: uniqueness_score(df2[col]) for col in df2.columns}
    
    # Find potential primary keys (highly unique columns)
    potential_pk_cols = [col for col, score in df1_uniqueness.items() if score > 0.8]
    
    # If no high-uniqueness columns found, use all columns
    if not potential_pk_cols:
        potential_pk_cols = list(df1.columns)
    
    # Compare each potential primary key with columns in the second dataframe
    for pk_col in potential_pk_cols:
        # First check for exact name matches
        if pk_col in df2.columns:
            # Check data compatibility
            if df1[pk_col].dtype == df2[pk_col].dtype:
                # Calculate value overlap score
                df1_values = set(df1[pk_col].dropna().astype(str))
                df2_values = set(df2[pk_col].dropna().astype(str))
                
                if df1_values and df2_values:  # Ensure sets are not empty
                    overlap = len(df1_values.intersection(df2_values)) / min(len(df1_values), len(df2_values))
                    
                    # Higher score for exact name match with good overlap
                    score = 0.7 * 100 + 0.3 * (overlap * 100)
                    suggested_keys.append((pk_col, pk_col, score))
        
        # For fuzzy matching, only if fuzzywuzzy is available
        if FUZZYWUZZY_AVAILABLE:
            # For fuzzy matching, compare with all columns in df2
            for fk_col in df2.columns:
                # Skip exact matches (already handled)
                if pk_col == fk_col:
                    continue
                
                # Calculate string similarity between column names
                name_similarity = fuzz.ratio(pk_col.lower(), fk_col.lower())
                
                # If column names are similar enough
                if name_similarity >= threshold:
                    # Check if data types are compatible for joining
                    pk_type = str(df1[pk_col].dtype)
                    fk_type = str(df2[fk_col].dtype)
                    
                    # Consider compatible if same type or both numeric or both string-like
                    type_compatible = (
                        pk_type == fk_type or
                        (('int' in pk_type or 'float' in pk_type) and ('int' in fk_type or 'float' in fk_type)) or
                        (('object' in pk_type or 'str' in pk_type) and ('object' in fk_type or 'str' in fk_type))
                    )
                    
                    if type_compatible:
                        # Calculate value overlap score
                        try:
                            df1_values = set(df1[pk_col].dropna().astype(str))
                            df2_values = set(df2[fk_col].dropna().astype(str))
                            
                            if df1_values and df2_values:  # Ensure sets are not empty
                                overlap = len(df1_values.intersection(df2_values)) / min(len(df1_values), len(df2_values))
                                
                                # Compute final score based on name similarity and value overlap
                                # Weigh name similarity higher for fuzzy matches
                                score = 0.7 * name_similarity + 0.3 * (overlap * 100)
                                
                                if score >= threshold:
                                    suggested_keys.append((pk_col, fk_col, score))
                        except:
                            # Skip if value comparison fails
                            pass
        else:
            # If fuzzywuzzy is not available, only use exact column matches
            pass
    
    # Sort by score in descending order
    suggested_keys.sort(key=lambda x: x[2], reverse=True)
    
    # Return top 3 suggestions
    return suggested_keys[:3]