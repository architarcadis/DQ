import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

def show_data_quality_assessment():
    """
    Display comprehensive data quality assessment tools and visualizations.
    """
    st.header("📊 Data Quality Assessment")
    
    # Check if data is loaded
    if not st.session_state.datasets:
        st.info("Please upload a dataset using the sidebar first.", icon="ℹ️")
        return
    
    # Dataset selection
    dataset_name = st.selectbox(
        "Select a dataset to analyze",
        options=list(st.session_state.datasets.keys()),
        index=0 if st.session_state.active_dataset is None else list(st.session_state.datasets.keys()).index(st.session_state.active_dataset),
        key="data_quality_dataset_select"
    )
    
    df = st.session_state.datasets[dataset_name]
    
    # Data Quality Assessment Introduction
    st.markdown("""
    ## Your Data Quality Story
    
    Data quality is the foundation of all trustworthy analysis. This assessment tells the complete story of your data's health, 
    following a narrative journey through key quality dimensions based on the DAMA framework.
    
    **Why this matters:**
    - High-quality data leads to reliable insights and better decisions
    - Understanding quality issues helps prioritize data cleaning efforts
    - Documenting data quality is essential for regulatory compliance and stakeholder trust
    
    We'll guide you through each dimension with clear visualizations and actionable recommendations.
    """)
    
    # DAMA Framework information with improved visuals
    with st.expander("About DAMA Data Quality Framework", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            ### DAMA Data Quality Framework
            
            This assessment follows the industry-standard DAMA (Data Management Association) framework, which defines six critical dimensions of data quality:
            
            1. **Completeness**: Proportion of stored data against potential 100% complete
            2. **Uniqueness**: Absence of duplicate records in the dataset
            3. **Timeliness**: Currency of data relative to time of use
            4. **Validity**: Conformity to syntax and domain constraints
            5. **Accuracy**: Correctness relative to real-world values
            6. **Consistency**: Agreement across related data elements
            
            Each dimension is evaluated separately and then combined for a comprehensive assessment.
            """)
        with col2:
            # Display a gauge chart for overall quality as a teaser
            if 'datasets' in st.session_state and st.session_state.datasets:
                df = st.session_state.datasets[dataset_name]
                quality_score = calculate_quality_score(df)
                fig = create_gauge_chart(quality_score, "Overall Quality", 0, 100)
                st.plotly_chart(fig, use_container_width=True)
    
    # PDF Report Generation Button with direct implementation
    st.markdown("### Generate Professional Report")
    st.markdown("""
    Create a comprehensive PDF report with executive summary, detailed analysis, and recommendations.
    Perfect for sharing with stakeholders or documenting data quality for compliance purposes.
    """)
    
    if st.button("📄 Generate Professional PDF Report", type="primary", help="Creates a downloadable PDF with complete data quality analysis and recommendations"):
        with st.spinner("Generating comprehensive professional data quality report..."):
            try:
                # Import libraries directly here to avoid any issues
                from datetime import datetime
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib import colors
                from reportlab.lib.units import inch
                import io
                
                # Generate PDF report directly here
                pdf_buffer = generate_data_quality_pdf_report(df, dataset_name)
                
                # Offer PDF for download with descriptive name
                st.success("Professional report generated successfully! Click below to download.")
                st.download_button(
                    label="📥 Download Professional PDF Report",
                    data=pdf_buffer,
                    file_name=f"Data_Quality_Assessment_{dataset_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_report",
                    help="Download a comprehensive professional PDF report with analysis and recommendations"
                )
                
                # Provide additional context
                st.info("This report contains a complete data quality assessment following DAMA framework principles with detailed recommendations for improving data quality.")
                
            except Exception as e:
                st.error(f"Error generating PDF report: {str(e)}")
                st.info("For detailed error information, please check the logs. Please make sure all required libraries are installed.")
    
    st.markdown("---")
    
    # Main tabs for different data quality aspects based on DAMA framework with better narrative flow
    st.markdown("### Quality Assessment Journey")
    quality_tabs = st.tabs([
        "Executive Summary", 
        "1. Completeness", 
        "2. Uniqueness", 
        "3. Validity", 
        "4. Accuracy",
        "5. Consistency",
        "6. Timeliness"
    ])
    
    # Tab 1: Overview
    with quality_tabs[0]:
        st.subheader("Dataset Overview")
        
        # Display basic dataset information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            memory_usage = df.memory_usage(deep=True).sum()
            if memory_usage > 1024 * 1024:
                memory_str = f"{memory_usage / (1024 * 1024):.2f} MB"
            else:
                memory_str = f"{memory_usage / 1024:.2f} KB"
            st.metric("Memory Usage", memory_str)
        
        # Data quality score
        quality_score = calculate_quality_score(df)
        
        st.subheader("Data Quality Score")
        
        # Create gauge chart for quality score
        fig = create_gauge_chart(quality_score, "Data Quality Score", 0, 100)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quality breakdown
        st.subheader("Quality Breakdown")
        
        # Calculate quality metrics
        completeness = 100 - (df.isnull().mean().mean() * 100)
        uniqueness = 100 - ((df.duplicated().sum() / len(df)) * 100)
        
        # Try to calculate consistency based on data types
        consistency = calculate_consistency_score(df)
        
        # Display quality metrics
        quality_metrics = pd.DataFrame({
            'Metric': ['Completeness', 'Uniqueness', 'Consistency', 'Overall'],
            'Score': [completeness, uniqueness, consistency, quality_score]
        })
        
        # Create horizontal bar chart
        fig = px.bar(
            quality_metrics, 
            x='Score', 
            y='Metric',
            orientation='h',
            color='Score',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        fig.update_layout(xaxis_title="Score (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        
        # Data summary
        st.subheader("Data Summary")
        
        # Calculate data type counts and convert data types to strings to prevent serialization issues
        dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        
        # Create data type distribution pie chart
        fig = px.pie(
            dtype_counts, 
            values='Count', 
            names='Data Type',
            title='Distribution of Column Data Types'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample data preview
        st.subheader("Data Sample")
        st.dataframe(df.head(5), use_container_width=True)
    
    # Tab 2: Missing Values
    with quality_tabs[1]:
        st.subheader("Missing Values Analysis")
        
        # Calculate missing values
        missing = df.isnull().sum().reset_index()
        missing.columns = ['Column', 'Missing Values']
        missing['Percentage'] = (missing['Missing Values'] / len(df)) * 100
        
        # Sort by missing values count
        missing = missing.sort_values('Missing Values', ascending=False)
        
        # Add description & suggestions
        st.markdown("""
        This analysis shows missing values across all columns in your dataset. High percentages of missing 
        values might require imputation strategies or column removal.
        """)
        
        # Display metrics
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        missing_percent = (total_missing / total_cells) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Missing Values", total_missing)
        with col2:
            st.metric("Percentage of Missing Values", f"{missing_percent:.2f}%")
        
        # Create horizontal bar chart for missing values
        if missing['Missing Values'].sum() > 0:
            missing_filtered = missing[missing['Missing Values'] > 0]
            if len(missing_filtered) > 0:
                fig = px.bar(
                    missing_filtered,
                    x='Missing Values',
                    y='Column',
                    orientation='h',
                    color='Percentage',
                    color_continuous_scale='Reds',
                    labels={'Missing Values': 'Count of Missing Values', 'Column': ''},
                    title='Columns with Missing Values'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display missing values table
                st.dataframe(missing_filtered, use_container_width=True)
                
                # Recommendations for handling missing values
                st.subheader("Recommendations")
                
                for _, row in missing_filtered.iterrows():
                    if row['Percentage'] > 50:
                        st.warning(f"Column '{row['Column']}' is missing more than 50% of its values. Consider dropping this column.")
                    elif row['Percentage'] > 20:
                        st.info(f"Column '{row['Column']}' has {row['Percentage']:.1f}% missing values. Consider imputation techniques.")
                
                # Missing values pattern visualization
                st.subheader("Missing Values Pattern")
                
                # Create missing values heatmap
                try:
                    # Use Matplotlib for this specific visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df[missing_filtered['Column']].isnull(), cmap='viridis', yticklabels=False, cbar=False, ax=ax)
                    plt.title('Missing Values Pattern (Yellow = Missing)')
                    plt.xlabel('Columns')
                    plt.ylabel('Rows')
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not generate missing values heatmap: {str(e)}")
            else:
                st.success("No missing values found in this dataset!")
        else:
            st.success("No missing values found in this dataset!")
    
    # Tab 3: Distributions
    with quality_tabs[2]:
        st.subheader("Data Distributions")
        
        # Organize columns by data type
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Numeric distributions
        if numeric_columns:
            st.markdown("### Numeric Columns")
            
            # Column selector for numeric columns
            selected_numeric = st.multiselect(
                "Select numeric columns to visualize distributions",
                options=numeric_columns,
                default=numeric_columns[:min(3, len(numeric_columns))]
            )
            
            if selected_numeric:
                # Distribution charts
                for col in selected_numeric:
                    st.markdown(f"#### Distribution of {col}")
                    
                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{df[col].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{df[col].median():.2f}")
                    with col3:
                        st.metric("Min", f"{df[col].min():.2f}")
                    with col4:
                        st.metric("Max", f"{df[col].max():.2f}")
                    
                    # Create histogram and KDE
                    fig = px.histogram(
                        df, 
                        x=col,
                        marginal="box",
                        histnorm='percent',
                        title=f'Distribution of {col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical distributions
        if categorical_columns:
            st.markdown("### Categorical Columns")
            
            # Column selector for categorical columns
            selected_categorical = st.multiselect(
                "Select categorical columns to visualize distributions",
                options=categorical_columns,
                default=categorical_columns[:min(3, len(categorical_columns))]
            )
            
            if selected_categorical:
                # Distribution charts
                for col in selected_categorical:
                    st.markdown(f"#### Distribution of {col}")
                    
                    # Calculate value counts and percentage
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['Value', 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / value_counts['Count'].sum()) * 100
                    
                    # Display number of unique values
                    st.metric("Unique Values", df[col].nunique())
                    
                    # Truncate if too many unique values
                    max_display = 20
                    if len(value_counts) > max_display:
                        st.warning(f"Too many unique values ({len(value_counts)}). Showing top {max_display} values.")
                        value_counts = value_counts.head(max_display)
                    
                    # Create bar chart
                    fig = px.bar(
                        value_counts,
                        x='Value',
                        y='Count',
                        color='Count',
                        color_continuous_scale='Blues',
                        title=f'Distribution of {col}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Outliers
    with quality_tabs[3]:
        st.subheader("Outlier Detection")
        
        # Only analyze numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if not numeric_columns:
            st.info("No numeric columns found for outlier detection.")
        else:
            # Column selector for outlier detection
            selected_column = st.selectbox(
                "Select a column to detect outliers",
                options=numeric_columns
            )
            
            # Calculate outliers using IQR method
            Q1 = df[selected_column].quantile(0.25)
            Q3 = df[selected_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[selected_column] < lower_bound) | (df[selected_column] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            # Display outlier metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Outlier Count", outlier_count)
            with col2:
                st.metric("Outlier Percentage", f"{outlier_percentage:.2f}%")
            with col3:
                st.metric("IQR", f"{IQR:.2f}")
            
            # Display boxplot with outliers
            fig = px.box(
                df,
                y=selected_column,
                title=f'Boxplot of {selected_column} with Outliers'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display scatter plot with outliers highlighted
            if len(df) > 1000:
                sample_size = 1000
                st.info(f"Dataset is large. Showing a random sample of {sample_size} points in the scatter plot.")
                scatter_df = df.sample(sample_size)
            else:
                scatter_df = df
            
            # Create a column to identify outliers
            scatter_df['is_outlier'] = (scatter_df[selected_column] < lower_bound) | (scatter_df[selected_column] > upper_bound)
            
            fig = px.scatter(
                scatter_df,
                y=selected_column,
                x=scatter_df.index,
                color='is_outlier',
                color_discrete_map={True: 'red', False: 'blue'},
                title=f'Outlier Detection for {selected_column}'
            )
            fig.update_layout(xaxis_title="Index", yaxis_title=selected_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display outlier rows
            if outlier_count > 0:
                st.subheader("Outlier Rows")
                st.dataframe(outliers.sort_values(by=selected_column, ascending=False), use_container_width=True)
                
                # Recommendations for handling outliers
                st.subheader("Recommendations")
                
                if outlier_percentage > 10:
                    st.warning(f"High percentage of outliers ({outlier_percentage:.2f}%). This may indicate a skewed distribution rather than true outliers.")
                else:
                    st.info("Consider these options for handling outliers:")
                    st.markdown("""
                    - **Remove outliers**: If they represent errors or anomalies
                    - **Transform data**: Apply log or other transformations to reduce impact
                    - **Cap outliers**: Replace with min/max thresholds
                    - **Separate analysis**: Analyze outliers separately if they represent important rare cases
                    """)
    
    # Tab 5: Correlations
    with quality_tabs[4]:
        st.subheader("Correlation Analysis")
        
        # Only analyze numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_columns) < 2:
            st.info("Need at least 2 numeric columns to perform correlation analysis.")
        else:
            # Calculate correlation matrix
            corr_matrix = df[numeric_columns].corr()
            
            # Display heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title='Correlation Matrix Heatmap'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            st.subheader("Strong Correlations")
            
            # Extract strong correlations (absolute value > 0.5 and not self-correlation)
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
                
                # Display table of strong correlations
                st.dataframe(strong_corr_df, use_container_width=True)
                
                # Scatter plot for the strongest correlation
                if not strong_corr_df.empty:
                    top_corr = strong_corr_df.iloc[0]
                    var1, var2 = top_corr['Variable 1'], top_corr['Variable 2']
                    
                    st.subheader(f"Scatter Plot: {var1} vs {var2}")
                    
                    # Check if statsmodels is available for trendline
                    try:
                        import statsmodels.api as sm
                        has_statsmodels = True
                    except ImportError:
                        has_statsmodels = False
                        st.warning("Note: Statsmodels package is required for trendlines. Showing plot without trendline.")
                    
                    try:
                        # Use trendline only if statsmodels is available
                        if has_statsmodels:
                            fig = px.scatter(
                                df,
                                x=var1,
                                y=var2,
                                title=f'Correlation: {top_corr["Correlation"]:.2f}',
                                trendline='ols'
                            )
                        else:
                            fig = px.scatter(
                                df,
                                x=var1,
                                y=var2,
                                title=f'Correlation: {top_corr["Correlation"]:.2f}'
                            )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error creating scatter plot: {str(e)}")
                        # Show basic plot as fallback
                        st.scatter_chart(df[[var1, var2]].sample(min(1000, len(df))))
            else:
                st.info("No strong correlations (|r| > 0.5) found in the dataset.")
    
    # Tab 6: Duplicates
    with quality_tabs[5]:
        st.subheader("Duplicate Records Analysis")
        
        # Calculate duplicates
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        # Display duplicate metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Duplicate Records", duplicate_count)
        with col2:
            st.metric("Percentage", f"{duplicate_percentage:.2f}%")
        
        # Display duplicate rows if any
        if duplicate_count > 0:
            st.subheader("Duplicate Rows")
            duplicate_rows = df[df.duplicated(keep='first')]
            st.dataframe(duplicate_rows, use_container_width=True)
            
            # Allow user to select columns to check duplicates
            st.subheader("Check Duplicates by Specific Columns")
            
            selected_columns = st.multiselect(
                "Select columns to check for duplicates",
                options=df.columns.tolist()
            )
            
            if selected_columns:
                column_duplicates = df.duplicated(subset=selected_columns, keep=False)
                column_duplicate_count = column_duplicates.sum()
                column_duplicate_percentage = (column_duplicate_count / len(df)) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Duplicates by Selected Columns", column_duplicate_count)
                with col2:
                    st.metric("Percentage", f"{column_duplicate_percentage:.2f}%")
                
                if column_duplicate_count > 0:
                    # Group and count the duplicates
                    duplicate_groups = df[column_duplicates].groupby(selected_columns).size().reset_index(name='Count')
                    duplicate_groups = duplicate_groups.sort_values('Count', ascending=False)
                    
                    st.subheader("Duplicate Groups")
                    st.dataframe(duplicate_groups, use_container_width=True)
                    
                    # Show the actual duplicate rows
                    st.subheader("Rows with Duplicate Values")
                    duplicate_rows = df[column_duplicates].sort_values(by=selected_columns)
                    st.dataframe(duplicate_rows, use_container_width=True)
                else:
                    st.success("No duplicates found based on the selected columns!")
        else:
            st.success("No duplicate rows found in this dataset!")
    
    # Tab 7: Data Types
    with quality_tabs[6]:
        st.subheader("Data Types Analysis")
        
        # Get data types and convert to strings to avoid JSON serialization issues
        dtype_df = pd.DataFrame({
            'Column': df.columns.tolist(),
            'Data Type': df.dtypes.astype(str).tolist()
        })
        
        # Add number of unique values
        dtype_df['Unique Values'] = [df[col].nunique() for col in df.columns]
        
        # Add sample values
        dtype_df['Sample Values'] = [str(df[col].dropna().sample(min(3, df[col].count())).tolist()) for col in df.columns]
        
        # Display data types table
        st.dataframe(dtype_df, use_container_width=True)
        
        # Type conversion suggestions
        st.subheader("Type Conversion Suggestions")
        
        suggestions = []
        
        # Check for potential datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            # Skip columns with too many unique values (unlikely to be dates)
            if df[col].nunique() > min(100, len(df) * 0.5):
                continue
                
            # Try to parse as datetime
            try:
                sample = df[col].dropna().iloc[0]
                pd.to_datetime(sample)
                suggestions.append({
                    'Column': col,
                    'Current Type': 'object',
                    'Suggested Type': 'datetime',
                    'Reason': f"Values like '{sample}' appear to be dates/times"
                })
            except:
                pass
        
        # Check for potential categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.1 and df[col].nunique() <= 50:
                suggestions.append({
                    'Column': col,
                    'Current Type': 'object',
                    'Suggested Type': 'category',
                    'Reason': f"Only {df[col].nunique()} unique values ({unique_ratio:.1%} of data)"
                })
        
        # Check for potential numeric columns
        for col in df.select_dtypes(include=['object']).columns:
            # Skip columns already identified as potential datetime or categorical
            if any(s['Column'] == col for s in suggestions):
                continue
                
            # Try to convert to numeric
            try:
                # Remove common non-numeric characters and try to convert
                sample = df[col].dropna().iloc[0].replace('$', '').replace(',', '').strip()
                float(sample)
                suggestions.append({
                    'Column': col,
                    'Current Type': 'object',
                    'Suggested Type': 'numeric',
                    'Reason': f"Values like '{df[col].iloc[0]}' could be converted to numbers"
                })
            except:
                pass
        
        if suggestions:
            suggestions_df = pd.DataFrame(suggestions)
            st.dataframe(suggestions_df, use_container_width=True)
        else:
            st.info("No type conversion suggestions found.")

def calculate_quality_score(df):
    """
    Calculate an overall data quality score based on various quality dimensions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    Returns:
    --------
    float
        A score between 0 and 100 representing overall data quality
    """
    try:
        # Completeness (percentage of non-missing values)
        completeness = float(100 - (df.isnull().mean().mean() * 100))
        
        # Uniqueness (percentage of non-duplicate rows)
        uniqueness = float(100 - ((df.duplicated().sum() / len(df)) * 100))
        
        # Consistency score
        consistency = float(calculate_consistency_score(df))
        
        # Overall score (weighted average)
        # We could adjust weights based on importance of each dimension
        weights = {
            'completeness': 0.4,
            'uniqueness': 0.3,
            'consistency': 0.3
        }
        
        quality_score = float(
            weights['completeness'] * completeness +
            weights['uniqueness'] * uniqueness +
            weights['consistency'] * consistency
        )
        
        return round(float(quality_score), 1)
    except Exception as e:
        # Return a default score if calculation fails
        print(f"Error calculating quality score: {str(e)}")
        return 70.0

def calculate_consistency_score(df):
    """
    Calculate a consistency score based on data type validation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
        
    Returns:
    --------
    float
        A score between 0 and 100 representing data consistency
    """
    consistency_scores = []
    
    # Check numeric columns for range consistency
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        try:
            # Check for outliers using z-score
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            # Convert to Python native types to avoid serialization issues
            outlier_percentage = float((z_scores > 3).mean() * 100)
            # Higher outlier percentage = lower consistency
            col_consistency = float(100 - min(outlier_percentage * 5, 100))  # Scale issue impact
            consistency_scores.append(col_consistency)
        except Exception as e:
            # If any calculation fails, use a default value
            consistency_scores.append(70.0)
    
    # Check categorical/object columns for consistency
    for col in df.select_dtypes(include=['object', 'category']).columns:
        try:
            # Calculate entropy (higher entropy = more variation = potentially less consistency)
            value_counts = df[col].value_counts(normalize=True)
            # Skip columns with too many unique values
            if len(value_counts) > 100:
                continue
            
            # Check if values follow patterns expected for their type
            # Example: if it's likely a date column, check date formats are consistent
            # This is a simplified approach
            col_consistency = float(100 - min(((df[col].nunique() / len(df)) * 100), 100))
            consistency_scores.append(col_consistency)
        except Exception as e:
            # If any calculation fails, use a default value
            consistency_scores.append(70.0)
    
    # If no scores calculated, return a neutral score
    if not consistency_scores:
        return 70.0
        
    return round(float(sum(consistency_scores) / len(consistency_scores)), 1)

def create_gauge_chart(value, title, min_val=0, max_val=100):
    """
    Create a gauge chart for visualizing a score.
    
    Parameters:
    -----------
    value : float
        The value to display on the gauge
    title : str
        The title for the gauge chart
    min_val : float
        Minimum value on the gauge
    max_val : float
        Maximum value on the gauge
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The gauge chart figure
    """
    # Define thresholds for color zones
    low_threshold = min_val + (max_val - min_val) * 0.33
    med_threshold = min_val + (max_val - min_val) * 0.67
    
    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'steps': [
                {'range': [min_val, low_threshold], 'color': "red"},
                {'range': [low_threshold, med_threshold], 'color': "yellow"},
                {'range': [med_threshold, max_val], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def generate_data_quality_pdf_report(df, dataset_name):
    """
    Generate a comprehensive data quality assessment PDF report with narrative and visuals.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    dataset_name : str
        The name of the dataset
    
    Returns:
    --------
    BytesIO
        PDF report as bytes
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics import renderPM
    from reportlab.graphics.widgets.markers import makeMarker
    import numpy as np
    import io
    import tempfile
    from datetime import datetime
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Create buffer for PDF
    buffer = io.BytesIO()
    
    # Create document with good margins
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title=f"Data Quality Report - {dataset_name}"
    )
    
    # Get stylesheet and create enhanced styles
    styles = getSampleStyleSheet()
    
    # Define custom colors
    dark_blue = colors.Color(0, 0, 0.5)  # RGB dark blue
    dark_slate_blue = colors.Color(0.28, 0.24, 0.55)  # RGB dark slate blue
    dark_green = colors.Color(0, 0.5, 0)  # RGB dark green
    dark_red = colors.Color(0.6, 0, 0)  # RGB dark red
    
    # Define light colors
    light_blue = colors.Color(0.7, 0.8, 0.9)  # RGB light blue
    light_green = colors.Color(0.7, 0.9, 0.7)  # RGB light green
    light_yellow = colors.Color(1, 1, 0.7)  # RGB light yellow
    light_coral = colors.Color(0.9, 0.6, 0.6)  # RGB light coral
    light_grey = colors.Color(0.83, 0.83, 0.83)  # RGB light grey
    light_steel_blue = colors.Color(0.7, 0.8, 0.9)  # RGB light steel blue
    beige_color = colors.Color(0.96, 0.96, 0.86)  # RGB beige
    
    # Define neutral colors
    grey_color = colors.Color(0.5, 0.5, 0.5)  # RGB grey
    black_color = colors.Color(0, 0, 0)  # RGB black
    white_color = colors.Color(1, 1, 1)  # RGB white
    
    # Define signal colors
    red_color = colors.Color(1, 0, 0)  # RGB red
    orange_color = colors.Color(1, 0.65, 0)  # RGB orange
    green_color = colors.Color(0, 0.8, 0)  # RGB green
    
    # Custom styles for a professional look
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles["Heading1"],
        alignment=1,  # Center alignment
        fontName="Helvetica-Bold",
        fontSize=18,
        spaceAfter=10,
        textColor=dark_blue
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        spaceAfter=8,
        textColor=dark_blue,
        borderWidth=0,
        borderPadding=5,
        borderColor=dark_blue,
        borderRadius=2
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=dark_slate_blue,
        spaceBefore=10,
        spaceAfter=6
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceBefore=6,  # Add space before paragraphs to prevent overlaps
        spaceAfter=6    # Add space after paragraphs to prevent overlaps
    )
    
    emphasis_style = ParagraphStyle(
        'Emphasis',
        parent=normal_style,
        fontName="Helvetica-Bold",
        textColor=dark_blue,
        leading=15  # Increased line spacing for emphasis text
    )
    
    recommendation_style = ParagraphStyle(
        'Recommendation',
        parent=normal_style,
        fontName="Helvetica",
        fontSize=10,
        leftIndent=20,
        textColor=dark_green,
        leading=15  # Increased line spacing for recommendations
    )
    
    warning_style = ParagraphStyle(
        'Warning',
        parent=normal_style,
        fontName="Helvetica-Bold",
        textColor=dark_red,
        leading=15  # Increased line spacing for warnings
    )
    
    # Utility functions for creating visualizations
    def create_quality_pie_chart(data_dict, width=300, height=200):
        """Create a pie chart for data quality dimensions"""
        drawing = Drawing(width, height)
        
        # Create pie chart
        pie = Pie()
        pie.x = width / 2
        pie.y = height / 2 - 10
        pie.width = min(width, height) - 40
        pie.height = min(width, height) - 40
        pie.data = [val for val in data_dict.values()]
        pie.labels = [key for key in data_dict.keys()]
        
        # Set nice colors
        pie.slices.strokeWidth = 0.5
        pie.slices[0].fillColor = light_blue
        pie.slices[1].fillColor = light_green
        pie.slices[2].fillColor = light_yellow
        if len(data_dict) > 3:
            pie.slices[3].fillColor = light_coral
        
        # Add shadow effect
        pie.slices.popout = 5
        pie.slices[0].popout = 10
        
        # Add to drawing
        drawing.add(pie)
        
        # Add a legend
        y_position = height - 30
        for i, (label, value) in enumerate(data_dict.items()):
            drawing.add(Rect(20, y_position, 10, 10, fillColor=pie.slices[i].fillColor, strokeColor=black_color, strokeWidth=0.5))
            drawing.add(String(35, y_position, f"{label}: {value:.1f}%", fontSize=8))
            y_position -= 15
            
        return drawing
    
    def create_bar_chart(categories, values, title, width=400, height=200, horizontal=True):
        """Create a bar chart"""
        drawing = Drawing(width, height)
        
        if horizontal:
            # Horizontal bar chart
            bc = HorizontalBarChart()
            bc.height = height - 50
            bc.width = width - 100
            bc.x = 60
            bc.y = 15
            bc.valueAxis.labels.fontSize = 8
            bc.valueAxis.labels.textAnchor = 'middle'
            bc.categoryAxis.labels.fontSize = 8
            bc.categoryAxis.labels.boxAnchor = 'ne'
            bc.categoryAxis.labels.dx = -10
            bc.categoryAxis.labels.dy = -2
            bc.categoryAxis.labels.angle = 0
            bc.categoryAxis.categoryNames = categories
            
            # Add data
            bc.data = [values]
            bc.bars[0].fillColor = light_blue
            bc.bars[0].strokeWidth = 0.5
            
            # Add value labels
            bc.barLabels.nudge = 7
            bc.barLabels.fontSize = 7
            bc.barLabelFormat = '%.1f'
            bc.barLabels.visible = True
        else:
            # Vertical bar chart
            bc = VerticalBarChart()
            bc.height = height - 50
            bc.width = width - 100
            bc.x = 50
            bc.y = 20
            bc.valueAxis.labels.fontSize = 8
            bc.valueAxis.labels.textAnchor = 'middle'
            bc.categoryAxis.labels.fontSize = 8
            bc.categoryAxis.labels.boxAnchor = 'n'
            bc.categoryAxis.labels.dx = 0
            bc.categoryAxis.labels.dy = -5
            bc.categoryAxis.labels.angle = 30
            bc.categoryAxis.categoryNames = categories
            
            # Add data
            bc.data = [values]
            bc.bars[0].fillColor = light_blue
            bc.bars[0].strokeWidth = 0.5
            
            # Add value labels
            bc.barLabels.nudge = 7
            bc.barLabels.fontSize = 7
            bc.barLabelFormat = '%.1f'
            bc.barLabels.visible = True
        
        # Add title
        drawing.add(String(width/2, height-10, title, fontSize=10, fontName="Helvetica-Bold", textAnchor="middle"))
        
        # Add to drawing
        drawing.add(bc)
        
        return drawing
    
    def create_gauge_chart_drawing(value, title, min_val=0, max_val=100, width=300, height=150):
        """Create a simpler gauge chart without requiring Arc component"""
        drawing = Drawing(width, height)
        
        # Background
        drawing.add(Rect(0, 0, width, height, fillColor=white_color, strokeColor=None))
        
        # Define gauge parameters
        gauge_width = width * 0.8
        gauge_height = 20
        gauge_x = (width - gauge_width) / 2
        gauge_y = height / 2
        
        # Draw gauge background (rectangle)
        drawing.add(Rect(gauge_x, gauge_y, gauge_width, gauge_height, 
                        fillColor=light_grey, strokeColor=grey_color))
        
        # Determine color based on value
        if value <= max_val * 0.33:
            color = red_color
        elif value <= max_val * 0.67:
            color = orange_color
        else:
            color = green_color
        
        # Calculate filled width based on value
        filled_width = gauge_width * (value - min_val) / (max_val - min_val)
        
        # Draw filled portion of gauge
        drawing.add(Rect(gauge_x, gauge_y, filled_width, gauge_height, 
                        fillColor=color, strokeColor=None))
        
        # Add value text
        drawing.add(String(width/2, gauge_y + 35, f"{value:.1f}%", 
                          fontSize=14, fontName="Helvetica-Bold", textAnchor="middle"))
        
        # Add title
        drawing.add(String(width/2, height - 15, title, 
                          fontSize=12, fontName="Helvetica-Bold", textAnchor="middle"))
        
        # Add min/max labels
        drawing.add(String(gauge_x, gauge_y - 15, f"{min_val}%", 
                          fontSize=8, textAnchor="start"))
        drawing.add(String(gauge_x + gauge_width, gauge_y - 15, f"{max_val}%", 
                          fontSize=8, textAnchor="end"))
        
        # Draw assessment text
        quality_text = get_assessment_text(value)
        drawing.add(String(width/2, gauge_y - 15, quality_text, 
                          fontSize=9, fontName="Helvetica", textAnchor="middle"))
        
        return drawing
    
    def create_matplotlib_figure_to_image(fig):
        """Convert a matplotlib figure to a reportlab Image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image(buf, width=6*inch, height=4*inch)
        return img
    
    # Create story (content)
    story = []
    
    # Calculate quality metrics for use throughout the report
    completeness = 100 - (df.isnull().mean().mean() * 100)
    uniqueness = 100 - ((df.duplicated().sum() / len(df)) * 100)
    consistency = calculate_consistency_score(df)
    quality_score = calculate_quality_score(df)
    
    # Cover page with graphic
    story.append(Spacer(1, 0.5*inch))
    
    # Create a visually appealing cover page with a gauge
    gauge_drawing = create_gauge_chart_drawing(quality_score, "Overall Data Quality", width=400, height=200)
    story.append(gauge_drawing)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<font size='20'>Data Quality Assessment</font>", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"<font size='16'>Professional Report</font>", title_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Dataset name on cover
    story.append(Paragraph(f"<font size='14'>Dataset: {dataset_name}</font>", title_style))
    story.append(Spacer(1, 0.25*inch))
    
    # Visual representation of quality dimensions
    dimensions_data = {
        "Completeness": completeness,
        "Uniqueness": uniqueness, 
        "Consistency": consistency,
    }
    story.append(create_quality_pie_chart(dimensions_data, width=400, height=200))
    
    # Date and organization
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y, %H:%M')}", styles["Italic"]))
    story.append(Paragraph("Powered by DAMA Framework", styles["Italic"]))
    
    # Add page break after cover
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", subtitle_style))
    story.append(Spacer(1, 0.1*inch))
    
    toc_items = [
        "1. Executive Summary",
        "2. Dataset Overview",
        "3. Quality Metrics",
        "4. Completeness Analysis",
        "5. Uniqueness Analysis",
        "6. Validity Analysis",
        "7. Detailed Findings",
        "8. Recommendations",
        "9. Conclusion"
    ]
    
    for item in toc_items:
        story.append(Paragraph(f"    • {item}", normal_style))
    
    story.append(PageBreak())
    
    # 1. Executive Summary
    story.append(Paragraph("1. Executive Summary", subtitle_style))
    
    # Create executive summary with clear narrative
    if quality_score >= 80:
        summary_text = f"""
        This dataset demonstrates <b>good overall quality</b> with a composite score of {quality_score:.1f}%. 
        Analysis across key dimensions reveals strong performance in 
        {'completeness' if completeness >= 80 else ''} 
        {'and uniqueness' if uniqueness >= 80 else ''}.
        """
        story.append(Paragraph(summary_text, normal_style))
        
        summary_text2 = f"""
        The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. 
        {'<font color="darkgreen">No significant data quality issues were detected.</font>' if quality_score >= 90 else 'Some minor quality issues were identified and are detailed in this report.'}
        """
        story.append(Paragraph(summary_text2, normal_style))
        
        strengths = []
        if completeness >= 80: strengths.append("Completeness")
        if uniqueness >= 80: strengths.append("Uniqueness")
        if consistency >= 80: strengths.append("Consistency")
        
        summary_text3 = f"""<b>Key strengths:</b> {', '.join(strengths) if strengths else 'None identified'}
        
        This dataset is suitable for analytical purposes with minimal preprocessing required.
        """
        story.append(Paragraph(summary_text3, normal_style))
    elif quality_score >= 60:
        summary_text = f"""
        This dataset demonstrates <b>moderate quality</b> with a composite score of {quality_score:.1f}%. 
        Analysis reveals certain quality issues that should be addressed before conducting advanced analytics.
        """
        story.append(Paragraph(summary_text, normal_style))
        
        summary_text2 = f"""
        The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
        {'<font color="orange">Several quality issues were identified that require attention.</font>'}
        """
        story.append(Paragraph(summary_text2, normal_style))
        
        areas = []
        if completeness < 70: areas.append("Completeness")
        if uniqueness < 70: areas.append("Uniqueness")
        if consistency < 70: areas.append("Consistency")
        
        summary_text3 = f"""<b>Areas for improvement:</b> {', '.join(areas) if areas else 'None identified'}
        
        This dataset requires some preprocessing before analysis to address identified quality issues.
        """
        story.append(Paragraph(summary_text3, normal_style))
    else:
        summary_text = f"""
        This dataset demonstrates <b>significant quality issues</b> with a low composite score of {quality_score:.1f}%.
        Substantial data quality problems were identified that require immediate attention.
        """
        story.append(Paragraph(summary_text, normal_style))
        
        summary_text2 = f"""
        The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.
        <font color="darkred">Major quality issues were detected that will impact analytical reliability.</font>
        """
        story.append(Paragraph(summary_text2, normal_style))
        
        areas = []
        if completeness < 60: areas.append("Completeness")
        if uniqueness < 60: areas.append("Uniqueness")
        if consistency < 60: areas.append("Consistency")
        
        summary_text3 = f"""<b>Critical areas for improvement:</b> {', '.join(areas) if areas else 'None identified'}
        
        Extensive data cleaning and validation are strongly recommended before this dataset is used for analysis.
        """
        story.append(Paragraph(summary_text3, normal_style))
    
    # Add visual quality gauge
    story.append(Spacer(1, 0.2*inch))
    
    # Create a horizontal quality score bar with clear visual indicators
    quality_categories = ["Completeness", "Uniqueness", "Consistency", "Overall"]
    quality_values = [completeness, uniqueness, consistency, quality_score]
    
    bar_chart = create_bar_chart(
        quality_categories, 
        quality_values, 
        "Quality Scores by Dimension",
        width=500, 
        height=200,
        horizontal=True
    )
    story.append(bar_chart)
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"<b>Overall Quality Score: {quality_score:.1f}%</b> - {get_assessment_text(quality_score)}", emphasis_style))
    
    story.append(PageBreak())
    
    # 2. Dataset Overview
    story.append(Paragraph("2. Dataset Overview", subtitle_style))
    
    # Dataset information
    story.append(Paragraph("Basic Dataset Information", section_style))
    
    overview_data = [
        ["Metric", "Value"],
        ["Rows", str(df.shape[0])],
        ["Columns", str(df.shape[1])],
        ["Memory Size", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"],
        ["Duplicate Rows", f"{df.duplicated().sum()} ({(df.duplicated().sum() / len(df) * 100):.2f}%)"],
        ["Missing Cells", f"{df.isnull().sum().sum()} ({(df.isnull().sum().sum() / df.size * 100):.2f}%)"]
    ]
    
    # Create a nicely formatted table
    overview_table = Table(overview_data, colWidths=[2.5*inch, 2.5*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.7, 0.7, 0.9)),  # Light steel blue
        ('TEXTCOLOR', (0, 0), (-1, 0), white_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.Color(0.94, 0.92, 0.84)),  # Beige
        ('GRID', (0, 0), (-1, -1), 0.5, grey_color)
    ]))
    
    story.append(overview_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Data Type Distribution
    story.append(Paragraph("Column Data Types", section_style))
    
    # Get data type distribution for visualization
    dtype_counts = df.dtypes.astype(str).value_counts()
    dtype_names = dtype_counts.index.tolist()
    dtype_values = dtype_counts.values.tolist()
    
    # Create data types bar chart
    dtype_chart = create_bar_chart(
        dtype_names,
        dtype_values,
        "Column Data Types Distribution",
        width=500,
        height=200,
        horizontal=False
    )
    story.append(dtype_chart)
    story.append(Spacer(1, 0.2*inch))
    
    # Create table for data types details
    dtype_data = [["Data Type", "Count", "Percentage"]]
    for dtype, count in zip(dtype_names, dtype_values):
        dtype_data.append([
            dtype, 
            str(count),
            f"{(count / df.shape[1] * 100):.1f}%"
        ])
    
    dtype_table = Table(dtype_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    dtype_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), light_steel_blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), white_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (1, 0), (2, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), beige_color),
        ('GRID', (0, 0), (-1, -1), 0.5, grey_color)
    ]))
    
    story.append(dtype_table)
    
    # Add basic statistics visualization if numeric columns exist
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols and len(numeric_cols) > 0:
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Numeric Column Statistics", section_style))
        
        # Create descriptive statistics
        desc_stats = df[numeric_cols[:min(3, len(numeric_cols))]].describe().round(2)
        
        # Create matplotlib figure for numeric column distributions
        try:
            plt.figure(figsize=(8, 4))
            for i, col in enumerate(numeric_cols[:min(3, len(numeric_cols))]):
                plt.subplot(1, min(3, len(numeric_cols)), i+1)
                plt.hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue')
                plt.title(f"{col} Distribution")
                plt.tight_layout()
            
            story.append(create_matplotlib_figure_to_image(plt.gcf()))
            plt.close()
        except Exception as e:
            story.append(Paragraph(f"Could not create distribution visualization: {str(e)}", normal_style))
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Note: A sample of the dataset is available in the online interface.", normal_style))
    
    story.append(PageBreak())
    
    # 3. Quality Metrics - DAMA Framework
    story.append(Paragraph("3. Quality Metrics (DAMA Framework)", subtitle_style))
    story.append(Paragraph("The DAMA framework provides a comprehensive approach to data quality assessment across six dimensions:", normal_style))
    
    # Create DAMA framework illustration
    story.append(Spacer(1, 0.2*inch))
    try:
        # Create a diagram using matplotlib
        plt.figure(figsize=(6, 4))
        plt.subplot(111, aspect='equal')
        
        # Create a circular diagram with 6 parts
        labels = ['Completeness', 'Uniqueness', 'Validity', 'Accuracy', 'Consistency', 'Timeliness']
        colors = ['#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0', '#B2912F', '#B276B2']
        
        plt.pie([1]*6, labels=labels, colors=colors, wedgeprops=dict(width=0.5, edgecolor='w'), startangle=90)
        plt.title("DAMA Data Quality Framework")
        
        # Convert to image and add to story
        story.append(create_matplotlib_figure_to_image(plt.gcf()))
        plt.close()
    except Exception as e:
        # If matplotlib fails, add a text description
        story.append(Paragraph("DAMA Framework Dimensions: Completeness, Uniqueness, Validity, Accuracy, Consistency, Timeliness", normal_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Create metric table with detailed explanations
    metrics_data = [
        ["Dimension", "Score", "Rating", "Description"],
        ["Completeness", f"{completeness:.1f}%", get_assessment_text(completeness), "Presence of required data values"],
        ["Uniqueness", f"{uniqueness:.1f}%", get_assessment_text(uniqueness), "Absence of duplicate records"],
        ["Validity", f"{consistency:.1f}%", get_assessment_text(consistency), "Conformity to data formats"],
        ["Accuracy", f"{min(completeness, uniqueness):.1f}%", get_assessment_text(min(completeness, uniqueness)), "How well data represents real-world"],
        ["Consistency", f"{consistency:.1f}%", get_assessment_text(consistency), "Agreement across data elements"],
        ["Overall Quality", f"{quality_score:.1f}%", get_assessment_text(quality_score), "Composite quality score"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[1.1*inch, 0.8*inch, 1.5*inch, 2.6*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), dark_blue),  # Using our custom dark_blue
        ('TEXTCOLOR', (0, 0), (-1, 0), white_color),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT'),
        ('ALIGN', (3, 1), (3, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, -1), (-1, -1), light_grey),
        ('GRID', (0, 0), (-1, -1), 0.5, grey_color)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Dimension explanations - shortened for readability
    dimension_explanations = """
    <b>• Completeness:</b> Measures extent of required data attributes. Higher completeness means fewer missing values.
    
    <b>• Uniqueness:</b> Measures absence of duplicate records. Higher uniqueness means fewer duplicate rows.
    
    <b>• Validity:</b> Assesses conformity to specified formats and domain constraints.
    
    <b>• Accuracy:</b> Evaluates how closely data values represent the true values of the attributes.
    
    <b>• Consistency:</b> Evaluates absence of contradictions across related data elements.
    
    <b>• Timeliness:</b> Assesses how current the data is relative to time of use.
    """
    
    story.append(Paragraph(dimension_explanations, normal_style))
    story.append(PageBreak())
    
    # 4. Completeness Analysis
    story.append(Paragraph("4. Completeness Analysis", subtitle_style))
    
    # Missing values analysis
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    
    # Add a visual gauge chart for completeness
    story.append(create_gauge_chart_drawing(completeness, "Completeness Score", width=400, height=150))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(f"Overall Completeness Score: {completeness:.1f}% ({get_assessment_text(completeness)})", emphasis_style))
    
    if len(missing_cols) > 0:
        story.append(Paragraph(f"Found {len(missing_cols)} columns with missing values out of {df.shape[1]} total columns.", normal_style))
        
        # Create visualization for missing values
        try:
            # Use matplotlib to create a missing values heatmap visualization
            plt.figure(figsize=(8, 4))
            
            # Create a heatmap of missing values for top columns
            cols_with_missing = missing_cols.index.tolist()[:min(10, len(missing_cols))]
            if cols_with_missing:
                missing_df = df[cols_with_missing].isnull()
                plt.imshow(missing_df.head(50).T, cmap='viridis', aspect='auto')
                plt.yticks(range(len(cols_with_missing)), cols_with_missing)
                plt.xlabel('Row Index')
                plt.ylabel('Column Name')
                plt.title('Missing Value Patterns (Yellow = Missing)')
                plt.colorbar(label='Missing')
                
                story.append(create_matplotlib_figure_to_image(plt.gcf()))
                plt.close()
        except Exception as e:
            story.append(Paragraph(f"Could not create missing values visualization: {str(e)}", normal_style))
        
        # Create a bar chart of missing percentages for top columns
        try:
            top_missing = missing_cols.sort_values(ascending=False)[:min(10, len(missing_cols))]
            percentages = [(count / len(df) * 100) for count in top_missing.values]
            
            bar_chart = create_bar_chart(
                top_missing.index.tolist(),
                percentages,
                "Columns with Most Missing Values (%)",
                width=500,
                height=250,
                horizontal=True
            )
            story.append(bar_chart)
        except Exception as e:
            story.append(Paragraph(f"Could not create missing values bar chart: {str(e)}", normal_style))
        
        # Create a table of columns with missing values
        missing_data = [["Column", "Missing Count", "Missing %", "Recommendation"]]
        
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            if percentage > 50:
                recommendation = "Consider removing this column"
            elif percentage > 20:
                recommendation = "Apply appropriate imputation"
            else:
                recommendation = "Minor issue - standard imputation"
                
            missing_data.append([
                col, 
                str(count),
                f"{percentage:.1f}%",
                recommendation
            ])
        
        # Add total row
        missing_data.append([
            "TOTAL", 
            str(missing_values.sum()),
            f"{(missing_values.sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%",
            ""
        ])
        
        missing_table = Table(missing_data, colWidths=[1.5*inch, 1*inch, 0.8*inch, 1.7*inch])
        missing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), light_steel_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (2, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, -1), (-1, -1), light_grey),
            ('GRID', (0, 0), (-1, -1), 0.5, grey_color)
        ]))
        
        story.append(missing_table)
        
        # Recommendations for handling missing values
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Completeness Recommendations:", section_style))
        
        if missing_values.sum() / df.size > 0.2:
            story.append(Paragraph("• The dataset has significant missing values that require attention before analysis.", warning_style))
        
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            if percentage > 50:
                story.append(Paragraph(f"• Column '{col}' has {percentage:.1f}% missing values. Consider removing this column as it may not provide reliable information.", recommendation_style))
            elif percentage > 20:
                story.append(Paragraph(f"• Column '{col}' has {percentage:.1f}% missing values. Consider using advanced imputation techniques like KNN or model-based imputation.", recommendation_style))
            elif percentage > 5:
                story.append(Paragraph(f"• Column '{col}' has {percentage:.1f}% missing values. Standard imputation methods like mean/median/mode replacement should be sufficient.", recommendation_style))
    else:
        story.append(Paragraph("No missing values detected in this dataset. Excellent completeness!", normal_style))
    
    story.append(PageBreak())
    
    # 5. Uniqueness Analysis
    story.append(Paragraph("5. Uniqueness Analysis", subtitle_style))
    
    # Duplicates analysis
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100
    
    # Add a visual gauge chart for uniqueness
    story.append(create_gauge_chart_drawing(uniqueness, "Uniqueness Score", width=400, height=150))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(f"Overall Uniqueness Score: {uniqueness:.1f}% ({get_assessment_text(uniqueness)})", emphasis_style))
    
    if duplicate_count > 0:
        story.append(Paragraph(f"Found {duplicate_count} duplicate rows ({duplicate_percentage:.2f}% of total rows).", normal_style))
        
        # Add a visual representation of duplicates
        try:
            plt.figure(figsize=(8, 4))
            labels = ['Unique Rows', 'Duplicate Rows']
            sizes = [len(df) - duplicate_count, duplicate_count]
            colors = ['#66b3ff', '#ff9999']
            explode = (0, 0.1)  # explode the 2nd slice (duplicates)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Proportion of Duplicate Rows')
            
            story.append(create_matplotlib_figure_to_image(plt.gcf()))
            plt.close()
        except Exception as e:
            story.append(Paragraph(f"Could not create duplicates visualization: {str(e)}", normal_style))
        
        # Potential ID columns analysis
        potential_id_cols = []
        for col in df.columns:
            if df[col].nunique() == len(df):
                potential_id_cols.append(col)
        
        if potential_id_cols:
            story.append(Paragraph("Potential unique identifier columns found:", normal_style))
            for col in potential_id_cols:
                story.append(Paragraph(f"• Column '{col}' has unique values for all rows", normal_style))
            
        # Recommendations for handling duplicates
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Uniqueness Recommendations:", section_style))
        
        if duplicate_percentage > 20:
            story.append(Paragraph("• The dataset has a high percentage of duplicate rows. This suggests a potential issue with data collection or processing.", warning_style))
            story.append(Paragraph("• Investigate the source of these duplicates before removing them as they may indicate a systemic issue.", recommendation_style))
        elif duplicate_percentage > 5:
            story.append(Paragraph("• The dataset has a moderate number of duplicate rows that should be addressed.", normal_style))
            story.append(Paragraph("• Consider using DataFrame.drop_duplicates() to remove duplicate rows, potentially preserving the first or last occurrence.", recommendation_style))
        else:
            story.append(Paragraph("• The dataset has a small number of duplicate rows that can be safely removed.", normal_style))
            story.append(Paragraph("• Use DataFrame.drop_duplicates() to clean the dataset before analysis.", recommendation_style))
    else:
        story.append(Paragraph("No duplicate rows detected in this dataset. Excellent uniqueness!", normal_style))
    
    # Handle potential composite keys
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Note on Composite Keys:", section_style))
    story.append(Paragraph("Even without a single unique identifier column, combinations of columns may form composite keys that uniquely identify rows.", normal_style))
    
    story.append(PageBreak())
    
    # 6. Validity Analysis
    story.append(Paragraph("6. Validity Analysis", subtitle_style))
    
    # Data type validity
    # Add a visual gauge chart for consistency/validity
    story.append(create_gauge_chart_drawing(consistency, "Validity Score", width=400, height=150))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph(f"Overall Validity Score: {consistency:.1f}% ({get_assessment_text(consistency)})", emphasis_style))
    
    # Create data type distribution visualization
    try:
        # Create a bar chart of data types
        data_types = df.dtypes.astype(str).value_counts()
        
        plt.figure(figsize=(8, 4))
        bars = plt.bar(data_types.index, data_types.values, color='skyblue')
        plt.xlabel('Data Type')
        plt.ylabel('Count')
        plt.title('Column Data Types Distribution')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height}',
                     ha='center', va='bottom')
        
        plt.tight_layout()
        story.append(create_matplotlib_figure_to_image(plt.gcf()))
        plt.close()
    except Exception as e:
        story.append(Paragraph(f"Could not create data types visualization: {str(e)}", normal_style))
    
    # Data type summary
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Data Type Distribution:", section_style))
    
    # Count of each data type
    dtypes = df.dtypes.astype(str)
    type_counts = dtypes.value_counts()
    
    dtype_data = []
    for dtype, count in type_counts.items():
        dtype_data.append(f"• {dtype}: {count} columns ({count/len(dtypes)*100:.1f}%)")
    
    for line in dtype_data:
        story.append(Paragraph(line, normal_style))
    
    # Validity checks
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("Format Validity:", section_style))
    
    # Check for non-standard data types or formats
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if object_cols:
        story.append(Paragraph(f"Found {len(object_cols)} columns with text/object data types that may require format validation:", normal_style))
        for col in object_cols[:5]:  # Limit to first 5 to avoid too much detail
            story.append(Paragraph(f"• Column '{col}' is stored as object/string type", normal_style))
        
        if len(object_cols) > 5:
            story.append(Paragraph(f"... and {len(object_cols) - 5} more object columns", normal_style))
        
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("Validity Recommendations:", section_style))
        story.append(Paragraph("• Consider validating text columns against expected formats (emails, URLs, codes, etc.)", recommendation_style))
        story.append(Paragraph("• For columns containing numeric data stored as strings, convert to appropriate numeric types", recommendation_style))
        story.append(Paragraph("• For columns representing dates, use pd.to_datetime() to convert to datetime format", recommendation_style))
    else:
        story.append(Paragraph("All columns use standard data types. No immediate validity concerns detected.", normal_style))
    
    story.append(PageBreak())
    
    # 7. Detailed Findings
    story.append(Paragraph("7. Detailed Findings", subtitle_style))
    
    # Summarize all findings
    story.append(Paragraph("Summary of Data Quality Issues:", section_style))
    
    # Create a comprehensive table of all issues
    findings = []
    
    # Completeness issues
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            if percentage > 5:  # Only report columns with >5% missing
                findings.append({
                    "Category": "Completeness",
                    "Issue": f"Missing values in '{col}'",
                    "Severity": "High" if percentage > 50 else "Medium" if percentage > 20 else "Low",
                    "Details": f"{count} values ({percentage:.1f}%)"
                })
    
    # Uniqueness issues
    if duplicate_count > 0:
        findings.append({
            "Category": "Uniqueness",
            "Issue": "Duplicate rows",
            "Severity": "High" if duplicate_percentage > 20 else "Medium" if duplicate_percentage > 5 else "Low",
            "Details": f"{duplicate_count} rows ({duplicate_percentage:.1f}%)"
        })
    
    # Validity issues (simplified)
    for col in object_cols[:3]:  # Limit to first 3
        findings.append({
            "Category": "Validity",
            "Issue": f"Potential format issue in '{col}'",
            "Severity": "Medium",
            "Details": "Text column may need validation"
        })
    
    # Create a visual representation of findings by category
    if findings:
        # Count issues by category and severity
        categories = {}
        severities = {"High": 0, "Medium": 0, "Low": 0}
        
        for finding in findings:
            cat = finding["Category"]
            sev = finding["Severity"]
            
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
            severities[sev] += 1
        
        # Create visualizations
        try:
            # Category distribution
            plt.figure(figsize=(10, 4))
            
            # First subplot - Categories
            plt.subplot(1, 2, 1)
            plt.bar(categories.keys(), categories.values(), color='skyblue')
            plt.title('Issues by Category')
            plt.xlabel('Category')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
            # Second subplot - Severities
            plt.subplot(1, 2, 2)
            bars = plt.bar(severities.keys(), severities.values(), 
                            color=['#ff6666', '#ffcc66', '#66cc66'])  # Red, Yellow, Green
            plt.title('Issues by Severity')
            plt.xlabel('Severity')
            plt.ylabel('Count')
            
            plt.tight_layout()
            story.append(create_matplotlib_figure_to_image(plt.gcf()))
            plt.close()
        except Exception as e:
            story.append(Paragraph(f"Could not create findings visualization: {str(e)}", normal_style))
        
        # If we have findings, create a table
        findings_data = [["Category", "Issue", "Severity", "Details"]]
        for finding in findings:
            findings_data.append([
                finding["Category"],
                finding["Issue"],
                finding["Severity"],
                finding["Details"]
            ])
        
        findings_table = Table(findings_data, colWidths=[1*inch, 2*inch, 0.8*inch, 1.2*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), dark_blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), white_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), beige_color),
            ('GRID', (0, 0), (-1, -1), 0.5, grey_color)
        ]))
        
        story.append(findings_table)
    else:
        story.append(Paragraph("No significant data quality issues were detected in this dataset.", normal_style))
    
    story.append(PageBreak())
    
    # 8. Recommendations
    story.append(Paragraph("8. Recommendations", subtitle_style))
    
    # Create actionable recommendations based on findings
    story.append(Paragraph("Based on our analysis, we recommend the following actions to improve data quality:", normal_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Create a visual representation of the recommendation workflow
    try:
        plt.figure(figsize=(8, 3))
        
        # Create a simple flowchart visualization
        plt.plot([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 0, 0, 1, 1, 0], 'k', linewidth=2)
        
        # Add boxes
        plt.text(0, 0.5, "Assess\nQuality", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", lw=1))
        plt.text(1, 0.5, "Clean\nData", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="green", lw=1))
        plt.text(2, 0.5, "Validate\nResults", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", lw=1))
        plt.text(3, 0.5, "Document\nProcess", ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", ec="red", lw=1))
        
        plt.title("Data Quality Improvement Workflow")
        plt.axis('off')
        plt.tight_layout()
        
        story.append(create_matplotlib_figure_to_image(plt.gcf()))
        plt.close()
    except Exception as e:
        story.append(Paragraph(f"Could not create workflow visualization: {str(e)}", normal_style))
    
    # General recommendations with improved styling and spacing
    general_recs = []
    
    # Completeness recommendations
    if df.isnull().sum().sum() > 0:
        general_recs.append(Paragraph("<b>1. Address Missing Values:</b>", section_style))
        
        # Add missing columns recommendations with separation
        cols_to_remove = [col for col, count in missing_cols.items() if count/len(df) > 0.5]
        if cols_to_remove:
            general_recs.append(Paragraph("• Remove columns with >50% missing values:", normal_style))
            for col in cols_to_remove:
                general_recs.append(Paragraph(f"  - {col} ({missing_cols[col]/len(df)*100:.1f}% missing)", recommendation_style))
        
        cols_to_impute = [col for col, count in missing_cols.items() if 0.1 < count/len(df) <= 0.5]
        if cols_to_impute:
            general_recs.append(Paragraph("• Apply advanced imputation for columns with 10-50% missing values:", normal_style))
            for col in cols_to_impute:
                general_recs.append(Paragraph(f"  - {col} ({missing_cols[col]/len(df)*100:.1f}% missing)", recommendation_style))
        
        cols_simple_impute = [col for col, count in missing_cols.items() if 0 < count/len(df) <= 0.1]
        if cols_simple_impute:
            general_recs.append(Paragraph("• Use simple imputation (mean/median/mode) for columns with <10% missing values:", normal_style))
            for col in cols_simple_impute[:5]:  # Limit to first 5
                general_recs.append(Paragraph(f"  - {col} ({missing_cols[col]/len(df)*100:.1f}% missing)", recommendation_style))
            if len(cols_simple_impute) > 5:
                general_recs.append(Paragraph(f"  - ... and {len(cols_simple_impute)-5} more columns", recommendation_style))
        
        general_recs.append(Spacer(1, 0.1*inch))
    
    # Uniqueness recommendations
    if duplicate_count > 0:
        general_recs.append(Paragraph("<b>2. Handle Duplicate Records:</b>", section_style))
        if duplicate_percentage > 20:
            general_recs.append(Paragraph("• Investigate the source of duplicate records as this level indicates a potential systematic issue", normal_style))
            general_recs.append(Paragraph("• Review data collection and integration processes", normal_style))
        general_recs.append(Paragraph("• Remove duplicate records using appropriate deduplication strategy", normal_style))
        general_recs.append(Paragraph("• Consider which duplicates to keep (first, last, or aggregate)", normal_style))
        general_recs.append(Spacer(1, 0.1*inch))
    
    # Validity recommendations
    if len(object_cols) > 0:
        general_recs.append(Paragraph("<b>3. Format Validation and Type Conversion:</b>", section_style))
        general_recs.append(Paragraph("• Validate text fields against expected formats or patterns", normal_style))
        general_recs.append(Paragraph("• Convert appropriate string columns to proper data types:", normal_style))
        general_recs.append(Paragraph("  - Numerical strings to int/float", recommendation_style))
        general_recs.append(Paragraph("  - Date strings to datetime", recommendation_style))
        general_recs.append(Paragraph("  - Boolean text ('yes'/'no', 'true'/'false') to boolean type", recommendation_style))
        general_recs.append(Spacer(1, 0.1*inch))
    
    # Add general best practices
    general_recs.append(Paragraph("<b>4. General Data Quality Best Practices:</b>", section_style))
    general_recs.append(Paragraph("• Document data cleaning steps for reproducibility", normal_style))
    general_recs.append(Paragraph("• Implement data validation rules for future data collection", normal_style))
    general_recs.append(Paragraph("• Maintain a data dictionary with clear column definitions and expected formats", normal_style))
    general_recs.append(Paragraph("• Establish regular data quality monitoring process", normal_style))
    
    # Add all recommendations to the story
    for rec in general_recs:
        story.append(rec)
    
    # Specific code snippets if appropriate
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("<b>5. Sample Code for Key Operations:</b>", section_style))
    
    code_examples = f"""
# Handling missing values
df_clean = df.dropna(axis=1, thresh=len(df)*0.5)  # Drop columns with >50% missing

# Removing duplicates
df_clean = df_clean.drop_duplicates(keep='first')

# Converting data types
# For date columns: df_clean['date_col'] = pd.to_datetime(df_clean['date_col'])
# For numeric columns: df_clean['num_col'] = pd.to_numeric(df_clean['num_col'], errors='coerce')

# Exporting cleaned data
df_clean.to_csv('cleaned_{dataset_name}.csv', index=False)
    """
    
    code_style = ParagraphStyle(
        'Code', 
        parent=normal_style, 
        fontName='Courier',
        fontSize=8,
        leading=10,
        leftIndent=20,
        rightIndent=20,
        backColor=light_grey,
        borderWidth=1,
        borderColor=grey_color,
        borderPadding=5,
        borderRadius=2
    )
    
    story.append(Paragraph(code_examples, code_style))
    
    story.append(PageBreak())
    
    # 9. Conclusion
    story.append(Paragraph("9. Conclusion", subtitle_style))
    
    # Summary visualization of all scores
    try:
        # Create radar chart for quality dimensions
        labels = ['Completeness', 'Uniqueness', 'Consistency', 'Validity', 'Overall']
        values = [completeness, uniqueness, consistency, consistency, quality_score]
        
        # Number of variables
        N = len(labels)
        
        # Create angles for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Values to plot (also close the loop)
        values += values[:1]
        
        # Create the plot
        plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, polar=True)
        
        # Draw one axis per variable and add labels
        plt.xticks(angles[:-1], labels, size=10)
        
        # Draw the chart
        ax.plot(angles, values, linewidth=2, linestyle='solid', label="Quality Scores")
        ax.fill(angles, values, alpha=0.25)
        
        # Set y-limits for consistency
        ax.set_ylim(0, 100)
        
        # Add a title
        plt.title("Data Quality Dimension Scores", size=14)
        
        # Add a legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        story.append(create_matplotlib_figure_to_image(plt.gcf()))
        plt.close()
    except Exception as e:
        story.append(Paragraph(f"Could not create radar chart: {str(e)}", normal_style))
    
    # Create a tailored conclusion with clear next steps
    story.append(Spacer(1, 0.2*inch))
    conclusion_text = f"""
    This comprehensive data quality assessment has evaluated the {dataset_name} dataset across the key dimensions 
    of the DAMA framework: Completeness, Uniqueness, Validity, Accuracy, Consistency, and Timeliness.
    """
    story.append(Paragraph(conclusion_text, normal_style))
    
    # Final quality score with visual emphasis
    quality_text = f"""
    <font size="12"><b>Overall Quality Score: {quality_score:.1f}%</b></font> ({get_assessment_text(quality_score)})
    """
    story.append(Paragraph(quality_text, emphasis_style))
    
    # Strengths and weaknesses
    strengths = [dim for dim, score in {'Completeness': completeness, 'Uniqueness': uniqueness, 'Consistency': consistency}.items() if score >= 80]
    weaknesses = [dim for dim, score in {'Completeness': completeness, 'Uniqueness': uniqueness, 'Consistency': consistency}.items() if score < 70]
    
    if strengths:
        story.append(Paragraph(f"<b>Strengths:</b> {', '.join(strengths)}", normal_style))
    
    if weaknesses:
        story.append(Paragraph(f"<b>Areas for Improvement:</b> {', '.join(weaknesses)}", normal_style))
    
    story.append(Spacer(1, 0.15*inch))
    
    # Next steps
    story.append(Paragraph("<b>Recommended Next Steps:</b>", section_style))
    
    next_steps = []
    
    if quality_score < 80:
        next_steps.append(Paragraph("1. Address the data quality issues identified in this report", normal_style))
        if weaknesses:
            next_steps.append(Paragraph(f"2. Focus on improving {', '.join(weaknesses)}", normal_style))
        next_steps.append(Paragraph("3. Implement the detailed recommendations provided in section 8", normal_style))
        next_steps.append(Paragraph("4. Re-evaluate data quality after implementing improvements", normal_style))
    else:
        next_steps.append(Paragraph("1. Maintain the current data quality processes", normal_style))
        next_steps.append(Paragraph("2. Proceed with analysis using the current dataset", normal_style))
        next_steps.append(Paragraph("3. Consider adding additional validation rules to maintain quality", normal_style))
        next_steps.append(Paragraph("4. Schedule regular data quality assessments to monitor ongoing quality", normal_style))
    
    for step in next_steps:
        story.append(step)
    
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("This report provides a foundation for data governance and quality management. By addressing the identified issues and implementing the recommendations, the dataset's quality and reliability for analytical purposes will be significantly improved.", normal_style))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d')} using DAMA Data Quality Framework", ParagraphStyle('Footer', parent=normal_style, fontSize=8, alignment=1)))
    
    # Build the PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer


def get_assessment_text(score):
    """Get qualitative assessment based on score"""
    if score >= 90:
        return "Excellent - Meets highest quality standards"
    elif score >= 80:
        return "Good - Suitable for most analyses"
    elif score >= 70:
        return "Satisfactory - Minor improvements needed"
    elif score >= 50:
        return "Needs improvement - Address issues before analysis"
    else:
        return "Poor - Significant data quality issues"


# PDF generation is now handled directly in the button click event above