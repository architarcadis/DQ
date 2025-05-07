import pandas as pd
import numpy as np
import streamlit as st
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Check if scikit-learn is available
try:
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def get_available_models(model_type):
    """
    Get the available models for a given model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('classification', 'regression', or 'clustering')
    
    Returns:
    --------
    list
        List of available model names
    """
    if not SKLEARN_AVAILABLE:
        return ["scikit-learn is required but not installed."]
        
    if model_type == 'classification':
        return [
            'Logistic Regression', 'Random Forest', 'Gradient Boosting', 
            'Decision Tree', 'K Neighbors', 'SVM'
        ]
    elif model_type == 'regression':
        return [
            'Linear Regression', 'Random Forest', 'Gradient Boosting',
            'Decision Tree', 'K Neighbors', 'Ridge', 'Lasso'
        ]
    elif model_type == 'clustering':
        return [
            'K-Means', 'Hierarchical', 'DBSCAN'
        ]
    else:
        return []

def get_model_instance(model_name, model_type):
    """
    Get a scikit-learn model instance based on the model name and type.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    model_type : str
        Type of model ('classification', 'regression', or 'clustering')
        
    Returns:
    --------
    model
        scikit-learn model instance
    """
    if not SKLEARN_AVAILABLE:
        return None
        
    model_mapping = {
        'classification': {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'K Neighbors': KNeighborsClassifier(),
            'SVM': SVC(probability=True)
        },
        'regression': {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'K Neighbors': KNeighborsRegressor(),
            'Ridge': Ridge(),
            'Lasso': Lasso()
        },
        'clustering': {
            'K-Means': KMeans(n_clusters=3),
            'Hierarchical': AgglomerativeClustering(n_clusters=3),
            'DBSCAN': DBSCAN()
        }
    }
    
    return model_mapping.get(model_type, {}).get(model_name)

def run_auto_model(df, target, features, model_type, model='auto', auto_model=True):
    """
    Run the automated modeling process using scikit-learn.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataset
    target : str or None
        Target variable name (None for clustering)
    features : list
        List of feature columns to use
    model_type : str
        Type of model ('classification', 'regression', or 'clustering')
    model : str
        Model name to use (if auto_model is False)
    auto_model : bool
        Whether to automatically select the best model
    
    Returns:
    --------
    dict
        Dictionary containing model results and information
    """
    if not SKLEARN_AVAILABLE:
        return {
            "error": "scikit-learn is required but not installed."
        }
        
    results = {}
    
    try:
        # Create working copy of the data
        data = df.copy()
        
        # Select only the relevant columns
        if model_type in ['classification', 'regression']:
            X = data[features]
            y = data[target]
        else:
            X = data[features]
            
        # Identify categorical features
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Split data for supervised learning
        if model_type in ['classification', 'regression']:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type in ['classification', 'regression']:
            if auto_model:
                # Try all models and select the best one
                available_models = get_available_models(model_type)
                best_score = -float('inf')
                best_model_name = None
                best_model = None
                
                for model_name in available_models:
                    model_instance = get_model_instance(model_name, model_type)
                    if model_instance:
                        # Create pipeline with preprocessing
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('model', model_instance)
                        ])
                        
                        # Cross-validation to evaluate model
                        if model_type == 'classification':
                            scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')
                        else:
                            scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
                        
                        avg_score = scores.mean()
                        if avg_score > best_score:
                            best_score = avg_score
                            best_model_name = model_name
                            best_model = model_instance
                
                if best_model:
                    # Create a fresh instance of the best model to avoid issues
                    best_model = get_model_instance(best_model_name, model_type)
                    
                    # Train the best model on all data
                    final_pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', best_model)
                    ])
                    final_pipeline.fit(X, y)
                    
                    # Make predictions for evaluation
                    y_pred = final_pipeline.predict(X_test)
                    
                    # Calculate metrics
                    if model_type == 'classification':
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted'),
                            'recall': recall_score(y_test, y_pred, average='weighted'),
                            'f1': f1_score(y_test, y_pred, average='weighted')
                        }
                        
                        # Create confusion matrix
                        classes = y.unique().tolist()
                        cm = np.zeros((len(classes), len(classes)))
                        for i, actual in enumerate(classes):
                            for j, pred in enumerate(classes):
                                cm[i, j] = sum((y_test == actual) & (y_pred == pred))
                    else:
                        metrics = {
                            'mse': mean_squared_error(y_test, y_pred),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2': r2_score(y_test, y_pred)
                        }
                    
                    # Calculate feature importance if available
                    feature_importance = []
                    # Get the model from the pipeline
                    model_step = final_pipeline.named_steps['model']
                    if hasattr(model_step, 'feature_importances_'):
                        # Get feature importance from the model
                        importance = model_step.feature_importances_
                        # Match with feature names
                        for i, feat in enumerate(features):
                            feature_importance.append({
                                'Feature': feat,
                                'Importance': importance[i] if i < len(importance) else 0
                            })
                    elif hasattr(model_step, 'coef_'):
                        # For linear models
                        coefs = model_step.coef_
                        if len(coefs.shape) > 1:
                            # For multi-class models
                            importance = np.abs(coefs).mean(axis=0)
                        else:
                            importance = np.abs(coefs)
                        
                        # Match with feature names
                        for i, feat in enumerate(features):
                            feature_importance.append({
                                'Feature': feat,
                                'Importance': importance[i] if i < len(importance) else 0
                            })
                    else:
                        # If feature importance is not available
                        for i, feat in enumerate(features):
                            feature_importance.append({
                                'Feature': feat,
                                'Importance': 1.0 / len(features)  # Uniform importance
                            })
                    
                    # Predictions on test data
                    pred_records = []
                    for i in range(len(X_test)):
                        pred_records.append({
                            'Actual': y_test.iloc[i],
                            'Predicted': y_pred[i]
                        })
                    
                    # Save the model to a buffer
                    model_buffer = io.BytesIO()
                    pickle.dump(final_pipeline, model_buffer)
                    model_buffer.seek(0)
                    
                    # Store results
                    results = {
                        'model_name': best_model_name,
                        'metrics': metrics,
                        'feature_importance': feature_importance,
                        'model_params': str(model_step.get_params()),
                        'model_binary': model_buffer.getvalue(),
                        'predictions': pd.DataFrame(pred_records)
                    }
                    
                    # Add confusion matrix for classification
                    if model_type == 'classification':
                        results['confusion_matrix'] = cm
                        results['classes'] = classes
                    
                    # Add cross-validation results
                    results['cv_results'] = scores.tolist()
            else:
                # Use specified model
                model_instance = get_model_instance(model, model_type)
                if model_instance:
                    # Create pipeline with preprocessing
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', model_instance)
                    ])
                    
                    # Train the model
                    pipeline.fit(X_train, y_train)
                    
                    # Make predictions for evaluation
                    y_pred = pipeline.predict(X_test)
                    
                    # Calculate metrics
                    if model_type == 'classification':
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted'),
                            'recall': recall_score(y_test, y_pred, average='weighted'),
                            'f1': f1_score(y_test, y_pred, average='weighted')
                        }
                        
                        # Create confusion matrix
                        classes = y.unique().tolist()
                        cm = np.zeros((len(classes), len(classes)))
                        for i, actual in enumerate(classes):
                            for j, pred in enumerate(classes):
                                cm[i, j] = sum((y_test == actual) & (y_pred == pred))
                    else:
                        metrics = {
                            'mse': mean_squared_error(y_test, y_pred),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'r2': r2_score(y_test, y_pred)
                        }
                    
                    # Calculate feature importance if available
                    feature_importance = []
                    if hasattr(model_instance, 'feature_importances_'):
                        # Get feature importance from the model
                        importance = model_instance.feature_importances_
                        # Match with feature names
                        for i, feat in enumerate(features):
                            feature_importance.append({
                                'Feature': feat,
                                'Importance': importance[i] if i < len(importance) else 0
                            })
                    elif hasattr(model_instance, 'coef_'):
                        # For linear models
                        coefs = model_instance.coef_
                        if len(coefs.shape) > 1:
                            # For multi-class models
                            importance = np.abs(coefs).mean(axis=0)
                        else:
                            importance = np.abs(coefs)
                        
                        # Match with feature names
                        for i, feat in enumerate(features):
                            feature_importance.append({
                                'Feature': feat,
                                'Importance': importance[i] if i < len(importance) else 0
                            })
                    else:
                        # If feature importance is not available
                        for i, feat in enumerate(features):
                            feature_importance.append({
                                'Feature': feat,
                                'Importance': 1.0 / len(features)  # Uniform importance
                            })
                    
                    # Predictions on test data
                    pred_records = []
                    for i in range(len(X_test)):
                        pred_records.append({
                            'Actual': y_test.iloc[i],
                            'Predicted': y_pred[i]
                        })
                    
                    # Save the model to a buffer
                    model_buffer = io.BytesIO()
                    pickle.dump(pipeline, model_buffer)
                    model_buffer.seek(0)
                    
                    # Store results
                    results = {
                        'model_name': model,
                        'metrics': metrics,
                        'feature_importance': feature_importance,
                        'model_params': str(model_instance.get_params()),
                        'model_binary': model_buffer.getvalue(),
                        'predictions': pd.DataFrame(pred_records)
                    }
                    
                    # Add confusion matrix for classification
                    if model_type == 'classification':
                        results['confusion_matrix'] = cm
                        results['classes'] = classes
                    
                    # Add cross-validation results
                    cv_scores = cross_val_score(pipeline, X, y, cv=5, 
                                               scoring='f1_weighted' if model_type == 'classification' 
                                               else 'neg_mean_squared_error')
                    results['cv_results'] = cv_scores.tolist()
                else:
                    results['error'] = f"Model {model} not found or not available."
        
        elif model_type == 'clustering':
            # Clustering models
            if model == 'auto' or model == 'K-Means':
                kmeans = KMeans(n_clusters=3)
                
                # Preprocess data
                X_processed = preprocessor.fit_transform(X)
                
                # Fit model
                kmeans.fit(X_processed)
                
                # Get cluster assignments
                clusters = kmeans.predict(X_processed)
                
                # Create cluster counts
                cluster_counts = {}
                for i in range(kmeans.n_clusters):
                    cluster_counts[f'Cluster {i}'] = sum(clusters == i)
                
                # Calculate silhouette score if available
                try:
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(X_processed, clusters)
                except:
                    silhouette = None
                
                # Store results
                results = {
                    'model_name': 'K-Means',
                    'metrics': {
                        'silhouette': silhouette,
                        'inertia': kmeans.inertia_,
                        'n_clusters': kmeans.n_clusters
                    },
                    'model_params': str(kmeans.get_params()),
                    'cluster_assignments': cluster_counts
                }
                
                # Save the model to a buffer
                model_buffer = io.BytesIO()
                pickle.dump((preprocessor, kmeans), model_buffer)
                model_buffer.seek(0)
                results['model_binary'] = model_buffer.getvalue()
                
                # Add the clusters to the dataframe
                cluster_df = df.copy()
                cluster_df['Cluster'] = clusters
                results['predictions'] = cluster_df[['Cluster']].head(20)
            else:
                model_instance = get_model_instance(model, model_type)
                if model_instance:
                    # Preprocess data
                    X_processed = preprocessor.fit_transform(X)
                    
                    # Fit model
                    model_instance.fit(X_processed)
                    
                    # Get cluster assignments
                    if hasattr(model_instance, 'labels_'):
                        clusters = model_instance.labels_
                    else:
                        clusters = model_instance.predict(X_processed)
                    
                    # Create cluster counts
                    cluster_counts = {}
                    unique_clusters = np.unique(clusters)
                    for i in unique_clusters:
                        # Handle noise points in DBSCAN (labeled as -1)
                        if i == -1:
                            cluster_counts['Noise'] = sum(clusters == i)
                        else:
                            cluster_counts[f'Cluster {i}'] = sum(clusters == i)
                    
                    # Calculate silhouette score if available
                    try:
                        from sklearn.metrics import silhouette_score
                        # Skip silhouette calculation if there's only one cluster or if there are noise points
                        if len(unique_clusters) > 1 and -1 not in unique_clusters:
                            silhouette = silhouette_score(X_processed, clusters)
                        else:
                            silhouette = None
                    except:
                        silhouette = None
                    
                    # Store results
                    results = {
                        'model_name': model,
                        'metrics': {
                            'silhouette': silhouette,
                            'n_clusters': len(cluster_counts)
                        },
                        'model_params': str(model_instance.get_params()),
                        'cluster_assignments': cluster_counts
                    }
                    
                    # Save the model to a buffer
                    model_buffer = io.BytesIO()
                    pickle.dump((preprocessor, model_instance), model_buffer)
                    model_buffer.seek(0)
                    results['model_binary'] = model_buffer.getvalue()
                    
                    # Add the clusters to the dataframe
                    cluster_df = df.copy()
                    cluster_df['Cluster'] = clusters
                    results['predictions'] = cluster_df[['Cluster']].head(20)
                else:
                    results['error'] = f"Model {model} not found or not available."
    
    except Exception as e:
        results['error'] = str(e)
    
    return results