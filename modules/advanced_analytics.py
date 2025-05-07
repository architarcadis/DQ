import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils.auto_model import run_auto_model, get_available_models

def reset_wizard():
    st.session_state.aa_step = 1
    st.session_state.aa_dataset = None
    st.session_state.aa_target = None
    st.session_state.aa_model_type = None
    st.session_state.aa_auto_model = True
    st.session_state.aa_model = None
    st.session_state.aa_features = None
    st.session_state.aa_results = None

def show_advanced_analytics():
    # Main content
    st.subheader("🤖 Advanced Analytics")
    st.markdown("Leverage machine learning to uncover patterns and make predictions")

    # Check if datasets exist
    if not st.session_state.datasets:
        st.info("👆 Start by uploading some data files using the sidebar", icon="ℹ️")
        
        # Using a stock photo for visual appeal
        st.image("https://pixabay.com/get/g2fc8c9695df8fbc6807e6fb9ad83232fe3105ff10ca2460a4edf2552c0df4cd9062742f29b4b553bf315b6c2b2766fb4b0acfab2b9495857454b5af151f0faae_1280.jpg", 
                caption="Advanced analytics with machine learning", 
                width=500)
        return

    # Initialize session state for the wizard
    if 'aa_step' not in st.session_state:
        st.session_state.aa_step = 1

    if 'aa_dataset' not in st.session_state:
        st.session_state.aa_dataset = None

    if 'aa_target' not in st.session_state:
        st.session_state.aa_target = None

    if 'aa_model_type' not in st.session_state:
        st.session_state.aa_model_type = None

    if 'aa_auto_model' not in st.session_state:
        st.session_state.aa_auto_model = True

    if 'aa_model' not in st.session_state:
        st.session_state.aa_model = None

    if 'aa_features' not in st.session_state:
        st.session_state.aa_features = None

    if 'aa_results' not in st.session_state:
        st.session_state.aa_results = None

    # Wizard progress bar
    progress = (st.session_state.aa_step - 1) / 4
    st.progress(progress)

    # Step 1: Choose dataset
    if st.session_state.aa_step == 1:
        st.subheader("Step 1: Choose Dataset")
        
        dataset_name = st.selectbox(
            "Select a dataset to analyze:",
            options=list(st.session_state.datasets.keys()),
            index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) 
                if st.session_state.active_dataset in st.session_state.datasets 
                else 0
        )
        
        if st.button("Next", use_container_width=True):
            st.session_state.aa_dataset = dataset_name
            st.session_state.aa_step = 2
            st.rerun()

    # Step 2: Select target variable
    elif st.session_state.aa_step == 2:
        st.subheader("Step 2: Select Target Variable")
        
        df = st.session_state.datasets[st.session_state.aa_dataset]
        
        # Option for clustering (no target)
        target_options = ["None (Clustering)"] + list(df.columns)
        
        target_var = st.selectbox(
            "Select the dependent variable (target):",
            options=target_options
        )
        
        if st.button("Next", use_container_width=True):
            if target_var == "None (Clustering)":
                st.session_state.aa_target = None
                st.session_state.aa_model_type = "clustering"
            else:
                st.session_state.aa_target = target_var
                
                # Determine model type based on target variable
                if df[target_var].dtype in [np.float64, np.int64]:
                    st.session_state.aa_model_type = "regression"
                else:
                    st.session_state.aa_model_type = "classification"
                    
            st.session_state.aa_step = 3
            st.rerun()
            
        if st.button("Back", key="back_step2"):
            st.session_state.aa_step = 1
            st.rerun()

    # Step 3: Model selection
    elif st.session_state.aa_step == 3:
        st.subheader("Step 3: Model Selection")
        
        # AutoModel toggle
        auto_model = st.toggle(
            "Use AutoModel",
            value=True,
            help="Automatically select the best model based on the data"
        )
        
        # If not AutoModel, show model selection
        if not auto_model:
            models = get_available_models(st.session_state.aa_model_type)
            selected_model = st.selectbox(
                "Select model:",
                options=models
            )
            st.session_state.aa_model = selected_model
        else:
            st.session_state.aa_model = "auto"
        
        st.session_state.aa_auto_model = auto_model
        
        if st.button("Next", use_container_width=True):
            st.session_state.aa_step = 4
            st.rerun()
            
        if st.button("Back", key="back_step3"):
            st.session_state.aa_step = 2
            st.rerun()

    # Step 4: Feature selection
    elif st.session_state.aa_step == 4:
        st.subheader("Step 4: Feature Selection")
        
        df = st.session_state.datasets[st.session_state.aa_dataset]
        
        # Get available features (exclude target)
        if st.session_state.aa_target:
            available_features = [c for c in df.columns if c != st.session_state.aa_target]
        else:
            available_features = list(df.columns)
        
        selected_features = st.multiselect(
            "Select features to include in the model:",
            options=available_features,
            default=available_features,
            help="Leave empty to use all available features"
        )
        
        # If no features selected, use all
        if not selected_features:
            selected_features = available_features
        
        st.session_state.aa_features = selected_features
        
        col1, col2 = st.columns(2)
        
        if col1.button("Run Analysis", use_container_width=True):
            with st.spinner("Running analysis... This may take a while."):
                try:
                    results = run_auto_model(
                        df=df,
                        target=st.session_state.aa_target,
                        features=selected_features,
                        model_type=st.session_state.aa_model_type,
                        model=st.session_state.aa_model,
                        auto_model=st.session_state.aa_auto_model
                    )
                    
                    st.session_state.aa_results = results
                    st.session_state.aa_step = 5
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
            
        if col2.button("Back", key="back_step4", use_container_width=True):
            st.session_state.aa_step = 3
            st.rerun()

    # Step 5: Results
    elif st.session_state.aa_step == 5:
        st.subheader("Results")
        
        if not st.session_state.aa_results:
            st.error("No results found. Please run the analysis again.")
            if st.button("Restart"):
                reset_wizard()
                st.rerun()
        else:
            results = st.session_state.aa_results
            
            # Create tabs for different result views
            tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Feature Importance", "Model Details", "Predictions"])
            
            with tab1:
                st.subheader("Model Performance Metrics")
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                else:
                    metrics_df = pd.DataFrame([results['metrics']])
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    if 'cv_results' in results:
                        st.subheader("Cross-Validation Results")
                        cv_results = results['cv_results']
                        if isinstance(cv_results, list):
                            cv_df = pd.DataFrame({'Score': cv_results})
                            st.dataframe(cv_df, use_container_width=True)
                            
                            # Plot CV results
                            fig = px.box(cv_df, y='Score', template="plotly_white", title="Cross-Validation Scores")
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Feature Importance")
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                elif 'feature_importance' in results:
                    importance_df = pd.DataFrame(results['feature_importance'])
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        template="plotly_white",
                        title="Feature Importance"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show SHAP plot if available
                    if 'shap_plot' in results:
                        st.subheader("SHAP Summary Plot")
                        st.pyplot(results['shap_plot'])
                else:
                    st.info("Feature importance not available for this model.")
            
            with tab3:
                st.subheader("Model Details")
                
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                else:
                    # Show model parameters
                    st.subheader("Model Parameters")
                    if 'model_params' in results:
                        st.code(results['model_params'])
                    
                    # Show pipeline if available
                    if 'pipeline' in results:
                        st.subheader("Processing Pipeline")
                        st.code(str(results['pipeline']))
                        
                    # Show model type and name
                    st.subheader("Model Information")
                    model_type = st.session_state.aa_model_type
                    st.write(f"Model Type: {model_type.capitalize() if model_type else 'Unknown'}")
                    st.write(f"Model: {results['model_name']}")
                    
                    # Download model button
                    if 'model_binary' in results:
                        st.download_button(
                            label="Download Model",
                            data=results['model_binary'],
                            file_name=f"arcadis_{st.session_state.aa_model_type}_{results['model_name']}.pkl",
                            mime="application/octet-stream"
                        )
            
            with tab4:
                st.subheader("Predictions")
                
                if 'error' in results:
                    st.error(f"Error: {results['error']}")
                elif 'predictions' in results:
                    pred_df = results['predictions']
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Plot residuals for regression
                    if st.session_state.aa_model_type == 'regression':
                        st.subheader("Residual Plot")
                        
                        if 'Actual' in pred_df.columns and 'Predicted' in pred_df.columns:
                            # Calculate residuals
                            pred_df['Residuals'] = pred_df['Actual'] - pred_df['Predicted']
                            
                            # Create scatter plot
                            fig = px.scatter(
                                pred_df,
                                x='Predicted',
                                y='Residuals',
                                template="plotly_white",
                                title="Residual Plot",
                                labels={"Predicted": "Predicted Values", "Residuals": "Residuals"}
                            )
                            
                            # Add horizontal line at y=0
                            fig.add_hline(y=0, line_dash="dash", line_color="#333333")
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Plot confusion matrix for classification
                    elif st.session_state.aa_model_type == 'classification' and 'confusion_matrix' in results:
                        st.subheader("Confusion Matrix")
                        
                        # Create heatmap
                        fig = px.imshow(
                            results['confusion_matrix'],
                            x=results['classes'],
                            y=results['classes'],
                            text_auto=True,
                            template="plotly_white",
                            title="Confusion Matrix",
                            color_continuous_scale=["#F5F5F5", "#F36F21"]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # Show cluster assignments for clustering
                    elif st.session_state.aa_model_type == 'clustering' and 'cluster_assignments' in results:
                        st.subheader("Cluster Distribution")
                        
                        cluster_df = pd.DataFrame({
                            'Cluster': list(results['cluster_assignments'].keys()),
                            'Count': list(results['cluster_assignments'].values())
                        })
                        
                        fig = px.bar(
                            cluster_df,
                            x='Cluster',
                            y='Count',
                            template="plotly_white",
                            title="Cluster Distribution"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Predictions not available for this model.")
            
            # Reset button
            if st.button("Start New Analysis", use_container_width=True):
                reset_wizard()
                st.rerun()