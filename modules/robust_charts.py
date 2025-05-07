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
from datetime import datetime, timedelta

def show_robust_charts():
    """
    Display a library of pre-populated robust charts and visualizations.
    """
    st.header("📈 Robust Charts & Visualizations")
    
    # Check if data is loaded
    if not st.session_state.datasets:
        st.info("Please upload a dataset using the sidebar first.", icon="ℹ️")
        return
    
    # Dataset selection
    dataset_name = st.selectbox(
        "Select a dataset to analyze",
        options=list(st.session_state.datasets.keys()),
        index=0 if st.session_state.active_dataset is None else list(st.session_state.datasets.keys()).index(st.session_state.active_dataset),
        key="robust_charts_dataset_select"
    )
    
    df = st.session_state.datasets[dataset_name]
    
    # Main tabs for different visualization categories
    chart_tabs = st.tabs([
        "Distributions", 
        "Time Series", 
        "Comparisons", 
        "Relationships",
        "Geospatial",
        "Custom Charts"
    ])
    
    # Tab 1: Distributions
    with chart_tabs[0]:
        st.subheader("Distribution Charts")
        
        st.markdown("""
        Distribution charts help you understand how values are spread across your data.
        Select different options to create the most appropriate visualization.
        """)
        
        distribution_type = st.selectbox(
            "Select distribution chart type",
            options=["Histogram", "Box Plot", "Violin Plot", "KDE Plot", "Bar Chart", "Pie Chart"]
        )
        
        if distribution_type in ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"]:
            # For charts that work with numerical data
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not numeric_columns:
                st.warning("No numeric columns found in this dataset.")
            else:
                columns = st.multiselect(
                    "Select columns to plot",
                    options=numeric_columns,
                    default=[numeric_columns[0]] if numeric_columns else []
                )
                
                if columns:
                    # Histogram
                    if distribution_type == "Histogram":
                        bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
                        
                        for col in columns:
                            fig = px.histogram(
                                df, 
                                x=col,
                                nbins=bins,
                                marginal="box",  # Show a boxplot at the margin
                                title=f'Histogram of {col}',
                                opacity=0.7,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Box Plot
                    elif distribution_type == "Box Plot":
                        group_by = st.selectbox(
                            "Group by (optional)",
                            options=["None"] + df.select_dtypes(include=['object', 'category']).columns.tolist()
                        )
                        
                        for col in columns:
                            if group_by != "None":
                                fig = px.box(
                                    df,
                                    x=group_by,
                                    y=col,
                                    title=f'Box Plot of {col} by {group_by}',
                                    color=group_by
                                )
                            else:
                                fig = px.box(
                                    df,
                                    y=col,
                                    title=f'Box Plot of {col}'
                                )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Violin Plot
                    elif distribution_type == "Violin Plot":
                        group_by = st.selectbox(
                            "Group by (optional)",
                            options=["None"] + df.select_dtypes(include=['object', 'category']).columns.tolist()
                        )
                        
                        for col in columns:
                            if group_by != "None":
                                fig = px.violin(
                                    df,
                                    x=group_by,
                                    y=col,
                                    title=f'Violin Plot of {col} by {group_by}',
                                    color=group_by,
                                    box=True  # Show box plot inside violin
                                )
                            else:
                                fig = px.violin(
                                    df,
                                    y=col,
                                    title=f'Violin Plot of {col}',
                                    box=True  # Show box plot inside violin
                                )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # KDE Plot
                    elif distribution_type == "KDE Plot":
                        # Use matplotlib/seaborn for KDE plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        for col in columns:
                            sns.kdeplot(df[col], fill=True, alpha=0.5, ax=ax, label=col)
                        
                        plt.title('Kernel Density Estimate (KDE) Plot')
                        plt.xlabel('Value')
                        plt.ylabel('Density')
                        plt.legend()
                        st.pyplot(fig)
        
        elif distribution_type in ["Bar Chart", "Pie Chart"]:
            # For charts that work well with categorical data
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if not categorical_columns:
                st.warning("No categorical columns found in this dataset.")
            else:
                column = st.selectbox(
                    "Select a column to plot",
                    options=categorical_columns
                )
                
                if column:
                    # Calculate value counts
                    value_counts = df[column].value_counts().reset_index()
                    value_counts.columns = ['Category', 'Count']
                    
                    # Limit to top N categories if there are too many
                    max_categories = st.slider("Maximum number of categories to show", 
                                              min_value=3, max_value=30, value=10)
                    
                    if len(value_counts) > max_categories:
                        other_count = value_counts.iloc[max_categories:]['Count'].sum()
                        value_counts = value_counts.iloc[:max_categories]
                        value_counts = pd.concat([
                            value_counts,
                            pd.DataFrame({'Category': ['Other'], 'Count': [other_count]})
                        ], ignore_index=True)
                    
                    # Bar Chart
                    if distribution_type == "Bar Chart":
                        # Sort options
                        sort_by = st.radio(
                            "Sort by",
                            options=["Count (Descending)", "Count (Ascending)", "Name (A-Z)", "Name (Z-A)"],
                            horizontal=True
                        )
                        
                        if sort_by == "Count (Descending)":
                            value_counts = value_counts.sort_values('Count', ascending=False)
                        elif sort_by == "Count (Ascending)":
                            value_counts = value_counts.sort_values('Count', ascending=True)
                        elif sort_by == "Name (A-Z)":
                            value_counts = value_counts.sort_values('Category', ascending=True)
                        elif sort_by == "Name (Z-A)":
                            value_counts = value_counts.sort_values('Category', ascending=False)
                        
                        fig = px.bar(
                            value_counts,
                            x='Category',
                            y='Count',
                            title=f'Bar Chart of {column}',
                            color='Count',
                            text='Count'
                        )
                        fig.update_traces(texttemplate='%{text}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Pie Chart
                    elif distribution_type == "Pie Chart":
                        # Choose between pie and donut
                        chart_style = st.radio(
                            "Chart style",
                            options=["Pie Chart", "Donut Chart"],
                            horizontal=True
                        )
                        
                        fig = px.pie(
                            value_counts,
                            values='Count',
                            names='Category',
                            title=f'{chart_style} of {column}'
                        )
                        
                        if chart_style == "Donut Chart":
                            fig.update_traces(hole=0.4)
                        
                        fig.update_traces(textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Time Series
    with chart_tabs[1]:
        st.subheader("Time Series Charts")
        
        st.markdown("""
        Time series charts help you visualize data over time and identify trends, seasonality, and anomalies.
        """)
        
        # Check if the dataset has date columns
        date_columns = []
        for col in df.columns:
            try:
                # Check if column can be converted to datetime
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                continue
        
        if not date_columns:
            st.warning("No valid date/time columns found in this dataset.")
            
            # Ask if user wants to create a demo time series
            if st.checkbox("Create a demo time series visualization"):
                create_demo_timeseries()
        else:
            # Time series chart options
            date_column = st.selectbox(
                "Select date/time column",
                options=date_columns
            )
            
            # Convert to datetime
            df_ts = df.copy()
            df_ts[date_column] = pd.to_datetime(df_ts[date_column])
            
            # Select numeric columns for the time series
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not numeric_columns:
                st.warning("No numeric columns found for time series visualization.")
            else:
                value_columns = st.multiselect(
                    "Select values to plot over time",
                    options=numeric_columns,
                    default=[numeric_columns[0]] if numeric_columns else []
                )
                
                if value_columns:
                    # Time series chart types
                    chart_type = st.selectbox(
                        "Select time series chart type",
                        options=["Line Chart", "Area Chart", "Bar Chart", "Candlestick", "Multi-axis Chart"]
                    )
                    
                    # Time aggregation
                    time_agg = st.selectbox(
                        "Aggregate time by",
                        options=["No Aggregation", "Day", "Week", "Month", "Quarter", "Year"]
                    )
                    
                    # Apply time aggregation if selected
                    if time_agg != "No Aggregation":
                        # Set date as index
                        df_ts = df_ts.set_index(date_column)
                        
                        # Aggregate based on selection
                        agg_dict = {col: 'mean' for col in value_columns}
                        
                        if time_agg == "Day":
                            df_ts = df_ts.resample('D').agg(agg_dict)
                        elif time_agg == "Week":
                            df_ts = df_ts.resample('W').agg(agg_dict)
                        elif time_agg == "Month":
                            df_ts = df_ts.resample('M').agg(agg_dict)
                        elif time_agg == "Quarter":
                            df_ts = df_ts.resample('Q').agg(agg_dict)
                        elif time_agg == "Year":
                            df_ts = df_ts.resample('Y').agg(agg_dict)
                        
                        # Reset index to have date as a column again
                        df_ts = df_ts.reset_index()
                    
                    # Create the requested chart
                    if chart_type == "Line Chart":
                        fig = px.line(
                            df_ts,
                            x=date_column,
                            y=value_columns,
                            title=f'Time Series: {", ".join(value_columns)} over {date_column}',
                            markers=True if len(df_ts) < 100 else False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Area Chart":
                        fig = px.area(
                            df_ts,
                            x=date_column,
                            y=value_columns,
                            title=f'Area Chart: {", ".join(value_columns)} over {date_column}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Bar Chart":
                        fig = px.bar(
                            df_ts,
                            x=date_column,
                            y=value_columns,
                            title=f'Bar Chart: {", ".join(value_columns)} over {date_column}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Candlestick":
                        # Need at least 4 values for candlestick (open, high, low, close)
                        if len(value_columns) < 4:
                            st.warning("Candlestick chart requires at least 4 numeric columns (OHLC).")
                        else:
                            ohlc_cols = value_columns[:4]
                            
                            fig = go.Figure(data=[go.Candlestick(
                                x=df_ts[date_column],
                                open=df_ts[ohlc_cols[0]],
                                high=df_ts[ohlc_cols[1]],
                                low=df_ts[ohlc_cols[2]],
                                close=df_ts[ohlc_cols[3]],
                                name="OHLC"
                            )])
                            
                            fig.update_layout(
                                title=f'Candlestick Chart: {", ".join(ohlc_cols)} over {date_column}',
                                xaxis_title=date_column,
                                yaxis_title="Value"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif chart_type == "Multi-axis Chart":
                        if len(value_columns) < 2:
                            st.warning("Multi-axis chart requires at least 2 numeric columns.")
                        else:
                            # Create multi-axis chart with two y-axes
                            y1_col = value_columns[0]
                            y2_col = value_columns[1]
                            
                            fig = make_subplots(specs=[[{"secondary_y": True}]])
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df_ts[date_column],
                                    y=df_ts[y1_col],
                                    name=y1_col,
                                    line=dict(color="blue")
                                ),
                                secondary_y=False
                            )
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df_ts[date_column],
                                    y=df_ts[y2_col],
                                    name=y2_col,
                                    line=dict(color="red")
                                ),
                                secondary_y=True
                            )
                            
                            fig.update_layout(
                                title=f'Multi-axis Chart: {y1_col} and {y2_col} over {date_column}',
                                xaxis_title=date_column
                            )
                            
                            fig.update_yaxes(title_text=y1_col, secondary_y=False)
                            fig.update_yaxes(title_text=y2_col, secondary_y=True)
                            
                            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Comparisons
    with chart_tabs[2]:
        st.subheader("Comparison Charts")
        
        st.markdown("""
        Comparison charts allow you to compare different categories, values, or metrics against each other.
        """)
        
        comparison_type = st.selectbox(
            "Select comparison chart type",
            options=["Grouped Bar Chart", "Stacked Bar Chart", "Radar Chart", "Parallel Categories", "Parallel Coordinates", "Sunburst"]
        )
        
        # Grouped and Stacked Bar Charts
        if comparison_type in ["Grouped Bar Chart", "Stacked Bar Chart"]:
            # Need at least one categorical and one numeric column
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if not categorical_columns or not numeric_columns:
                st.warning("This chart requires both categorical and numeric columns.")
            else:
                # Select categorical column for x-axis
                x_column = st.selectbox(
                    "Select category column (x-axis)",
                    options=categorical_columns
                )
                
                # Select numeric column for y-axis
                y_column = st.selectbox(
                    "Select value column (y-axis)",
                    options=numeric_columns
                )
                
                # Select categorical column for grouping/color
                color_column = st.selectbox(
                    "Select grouping column (color)",
                    options=["None"] + [col for col in categorical_columns if col != x_column]
                )
                
                if color_column == "None":
                    # Simple bar chart
                    fig = px.bar(
                        df,
                        x=x_column,
                        y=y_column,
                        title=f'{y_column} by {x_column}'
                    )
                else:
                    # Grouped or stacked bar chart
                    barmode = 'group' if comparison_type == "Grouped Bar Chart" else 'stack'
                    
                    fig = px.bar(
                        df,
                        x=x_column,
                        y=y_column,
                        color=color_column,
                        barmode=barmode,
                        title=f'{y_column} by {x_column}, grouped by {color_column}'
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Radar Chart
        elif comparison_type == "Radar Chart":
            # Need numeric columns for radar chart
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 3:
                st.warning("Radar chart requires at least 3 numeric columns.")
            else:
                # Select columns for radar chart
                radar_columns = st.multiselect(
                    "Select numeric columns for radar chart",
                    options=numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))]
                )
                
                # Select categorical column for grouping (optional)
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                group_column = st.selectbox(
                    "Select a column to compare groups (optional)",
                    options=["None"] + categorical_columns
                )
                
                if len(radar_columns) < 3:
                    st.warning("Please select at least 3 columns for the radar chart.")
                else:
                    if group_column != "None":
                        # Limit to top N groups
                        top_n = st.slider("Show top N groups", min_value=2, max_value=10, value=5)
                        
                        # Get top N groups by frequency
                        top_groups = df[group_column].value_counts().nlargest(top_n).index.tolist()
                        
                        # Filter dataframe to include only top groups
                        filtered_df = df[df[group_column].isin(top_groups)]
                        
                        # Create radar chart for each group
                        fig = go.Figure()
                        
                        for group in top_groups:
                            group_data = filtered_df[filtered_df[group_column] == group]
                            
                            # Calculate mean values for each metric
                            values = [group_data[col].mean() for col in radar_columns]
                            # Add first value at the end to close the loop
                            values.append(values[0])
                            
                            # All metrics plus the first one repeated
                            categories = radar_columns + [radar_columns[0]]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=categories,
                                fill='toself',
                                name=str(group)
                            ))
                    else:
                        # Create radar chart for entire dataset
                        fig = go.Figure()
                        
                        # Calculate mean values for each metric
                        values = [df[col].mean() for col in radar_columns]
                        # Add first value at the end to close the loop
                        values.append(values[0])
                        
                        # All metrics plus the first one repeated
                        categories = radar_columns + [radar_columns[0]]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name="Dataset"
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                            )
                        ),
                        title="Radar Chart"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Parallel Categories
        elif comparison_type == "Parallel Categories":
            # Need categorical columns
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_columns) < 2:
                st.warning("Parallel categories plot requires at least 2 categorical columns.")
            else:
                # Select categorical columns
                selected_columns = st.multiselect(
                    "Select categorical columns for parallel categories",
                    options=categorical_columns,
                    default=categorical_columns[:min(4, len(categorical_columns))]
                )
                
                # Select numeric column for color (optional)
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                color_column = st.selectbox(
                    "Select a numeric column for color (optional)",
                    options=["None"] + numeric_columns
                )
                
                if len(selected_columns) < 2:
                    st.warning("Please select at least 2 categorical columns.")
                else:
                    # Limit categorical values in each column
                    max_categories = 10
                    filtered_df = df.copy()
                    
                    for col in selected_columns:
                        value_counts = df[col].value_counts()
                        
                        if len(value_counts) > max_categories:
                            st.info(f"Column '{col}' has {len(value_counts)} unique values. Limiting to top {max_categories}.")
                            top_categories = value_counts.nlargest(max_categories).index.tolist()
                            filtered_df.loc[~filtered_df[col].isin(top_categories), col] = "Other"
                    
                    # Create parallel categories plot
                    if color_column != "None":
                        fig = px.parallel_categories(
                            filtered_df,
                            dimensions=selected_columns,
                            color=color_column,
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title="Parallel Categories Plot"
                        )
                    else:
                        fig = px.parallel_categories(
                            filtered_df,
                            dimensions=selected_columns,
                            title="Parallel Categories Plot"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Parallel Coordinates
        elif comparison_type == "Parallel Coordinates":
            # Need numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 3:
                st.warning("Parallel coordinates plot requires at least 3 numeric columns.")
            else:
                # Select numeric columns
                selected_columns = st.multiselect(
                    "Select numeric columns for parallel coordinates",
                    options=numeric_columns,
                    default=numeric_columns[:min(5, len(numeric_columns))]
                )
                
                # Select categorical column for color (optional)
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_column = st.selectbox(
                    "Select a categorical column for color (optional)",
                    options=["None"] + categorical_columns
                )
                
                if len(selected_columns) < 3:
                    st.warning("Please select at least 3 numeric columns.")
                else:
                    # Create parallel coordinates plot
                    if color_column != "None":
                        fig = px.parallel_coordinates(
                            df,
                            dimensions=selected_columns,
                            color=color_column,
                            title="Parallel Coordinates Plot"
                        )
                    else:
                        # If no color column selected, use the first column for color
                        fig = px.parallel_coordinates(
                            df,
                            dimensions=selected_columns,
                            color=selected_columns[0],
                            color_continuous_scale=px.colors.sequential.Viridis,
                            title="Parallel Coordinates Plot"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Sunburst Chart
        elif comparison_type == "Sunburst":
            # Need categorical columns for paths
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(categorical_columns) < 2:
                st.warning("Sunburst chart requires at least 2 categorical columns.")
            else:
                # Select path columns (hierarchical order)
                path_columns = st.multiselect(
                    "Select categorical columns for hierarchy (in order from center to outside)",
                    options=categorical_columns,
                    default=categorical_columns[:min(3, len(categorical_columns))]
                )
                
                # Select numeric column for values (optional)
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                value_column = st.selectbox(
                    "Select a numeric column for values (optional)",
                    options=["Count"] + numeric_columns
                )
                
                if len(path_columns) < 2:
                    st.warning("Please select at least 2 categorical columns.")
                else:
                    # Limit categorical values in each column
                    max_categories = 10
                    filtered_df = df.copy()
                    
                    for col in path_columns:
                        value_counts = df[col].value_counts()
                        
                        if len(value_counts) > max_categories:
                            st.info(f"Column '{col}' has {len(value_counts)} unique values. Limiting to top {max_categories}.")
                            top_categories = value_counts.nlargest(max_categories).index.tolist()
                            filtered_df.loc[~filtered_df[col].isin(top_categories), col] = "Other"
                    
                    # Create sunburst chart
                    if value_column == "Count":
                        fig = px.sunburst(
                            filtered_df,
                            path=path_columns,
                            title="Sunburst Chart"
                        )
                    else:
                        fig = px.sunburst(
                            filtered_df,
                            path=path_columns,
                            values=value_column,
                            title=f"Sunburst Chart (sized by {value_column})"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Relationships
    with chart_tabs[3]:
        st.subheader("Relationship Charts")
        
        st.markdown("""
        Relationship charts help you understand how different variables interact and influence each other.
        """)
        
        relationship_type = st.selectbox(
            "Select relationship chart type",
            options=["Scatter Plot", "Bubble Chart", "Heatmap", "Correlation Matrix", "Density Contour", "3D Scatter"]
        )
        
        # Scatter Plot
        if relationship_type == "Scatter Plot":
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.warning("Scatter plot requires at least 2 numeric columns.")
            else:
                # Select x and y columns
                x_column = st.selectbox(
                    "Select x-axis column",
                    options=numeric_columns
                )
                
                y_column = st.selectbox(
                    "Select y-axis column",
                    options=[col for col in numeric_columns if col != x_column],
                    index=0
                )
                
                # Select categorical column for color (optional)
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_column = st.selectbox(
                    "Select a categorical column for color (optional)",
                    options=["None"] + categorical_columns
                )
                
                # Add trendline option
                trendline = st.checkbox("Add trendline", value=True)
                
                # Check if statsmodels is available
                try:
                    import statsmodels.api as sm
                    has_statsmodels = True
                except ImportError:
                    has_statsmodels = False
                    if trendline:
                        st.warning("Note: Statsmodels package is required for trendlines. Showing plot without trendline.")
                
                # Create scatter plot
                if color_column != "None":
                    fig = px.scatter(
                        df,
                        x=x_column,
                        y=y_column,
                        color=color_column,
                        trendline="ols" if trendline and has_statsmodels else None,
                        title=f"Scatter Plot: {y_column} vs {x_column}"
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_column,
                        y=y_column,
                        trendline="ols" if trendline and has_statsmodels else None,
                        title=f"Scatter Plot: {y_column} vs {x_column}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Bubble Chart
        elif relationship_type == "Bubble Chart":
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 3:
                st.warning("Bubble chart requires at least 3 numeric columns.")
            else:
                # Select x, y, and size columns
                x_column = st.selectbox(
                    "Select x-axis column",
                    options=numeric_columns
                )
                
                y_column = st.selectbox(
                    "Select y-axis column",
                    options=[col for col in numeric_columns if col != x_column],
                    index=0
                )
                
                size_column = st.selectbox(
                    "Select bubble size column",
                    options=[col for col in numeric_columns if col not in [x_column, y_column]],
                    index=0
                )
                
                # Select categorical column for color (optional)
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_column = st.selectbox(
                    "Select a categorical column for color (optional)",
                    options=["None"] + categorical_columns
                )
                
                # Create bubble chart
                if color_column != "None":
                    fig = px.scatter(
                        df,
                        x=x_column,
                        y=y_column,
                        size=size_column,
                        color=color_column,
                        size_max=50,
                        title=f"Bubble Chart: {y_column} vs {x_column} (sized by {size_column})"
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_column,
                        y=y_column,
                        size=size_column,
                        size_max=50,
                        title=f"Bubble Chart: {y_column} vs {x_column} (sized by {size_column})"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        elif relationship_type == "Heatmap":
            # Need two categorical columns and one numeric column
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(categorical_columns) < 2 or not numeric_columns:
                st.warning("Heatmap requires at least 2 categorical columns and 1 numeric column.")
            else:
                # Select x and y categorical columns
                x_column = st.selectbox(
                    "Select x-axis column (categorical)",
                    options=categorical_columns
                )
                
                y_column = st.selectbox(
                    "Select y-axis column (categorical)",
                    options=[col for col in categorical_columns if col != x_column],
                    index=0
                )
                
                # Select numeric column for values
                value_column = st.selectbox(
                    "Select value column (numeric)",
                    options=numeric_columns
                )
                
                # Select aggregation method
                agg_method = st.selectbox(
                    "Select aggregation method",
                    options=["Mean", "Sum", "Count", "Median", "Min", "Max"]
                )
                
                # Map aggregation method to pandas function
                agg_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Count": "count",
                    "Median": "median",
                    "Min": "min",
                    "Max": "max"
                }
                
                # Create pivot table for heatmap
                pivot_df = df.pivot_table(
                    values=value_column,
                    index=y_column,
                    columns=x_column,
                    aggfunc=agg_map[agg_method]
                )
                
                # Limit categories if too many
                max_categories = 20  # Maximum categories to display
                if pivot_df.shape[0] > max_categories or pivot_df.shape[1] > max_categories:
                    st.warning(f"Too many categories to display. Limiting to top {max_categories}.")
                    
                    # Keep top categories by frequency
                    if pivot_df.shape[0] > max_categories:
                        top_rows = df[y_column].value_counts().nlargest(max_categories).index.tolist()
                        pivot_df = pivot_df.loc[pivot_df.index.isin(top_rows)]
                    
                    if pivot_df.shape[1] > max_categories:
                        top_cols = df[x_column].value_counts().nlargest(max_categories).index.tolist()
                        pivot_df = pivot_df.loc[:, pivot_df.columns.isin(top_cols)]
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df,
                    color_continuous_scale="Viridis",
                    title=f"Heatmap: {agg_method} of {value_column} by {x_column} and {y_column}",
                    labels=dict(x=x_column, y=y_column, color=f"{agg_method} of {value_column}")
                )
                
                fig.update_layout(
                    height=600,
                    xaxis=dict(tickangle=45),
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        elif relationship_type == "Correlation Matrix":
            # Need numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.warning("Correlation matrix requires at least 2 numeric columns.")
            else:
                # Select correlation method
                corr_method = st.selectbox(
                    "Select correlation method",
                    options=["Pearson", "Spearman", "Kendall"]
                )
                
                # Limit to selected columns
                selected_columns = st.multiselect(
                    "Select columns for correlation analysis",
                    options=numeric_columns,
                    default=numeric_columns
                )
                
                if len(selected_columns) < 2:
                    st.warning("Please select at least 2 numeric columns.")
                else:
                    # Calculate correlation matrix
                    corr_matrix = df[selected_columns].corr(method=corr_method.lower())
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto='.2f',
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        title=f"{corr_method} Correlation Matrix"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Density Contour
        elif relationship_type == "Density Contour":
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.warning("Density contour requires at least 2 numeric columns.")
            else:
                # Select x and y columns
                x_column = st.selectbox(
                    "Select x-axis column",
                    options=numeric_columns
                )
                
                y_column = st.selectbox(
                    "Select y-axis column",
                    options=[col for col in numeric_columns if col != x_column],
                    index=0
                )
                
                # Select categorical column for color (optional)
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_column = st.selectbox(
                    "Select a categorical column for color (optional)",
                    options=["None"] + categorical_columns
                )
                
                # Create density contour plot
                if color_column != "None":
                    fig = px.density_contour(
                        df,
                        x=x_column,
                        y=y_column,
                        color=color_column,
                        marginal_x="histogram",
                        marginal_y="histogram",
                        title=f"Density Contour: {y_column} vs {x_column}"
                    )
                else:
                    fig = px.density_contour(
                        df,
                        x=x_column,
                        y=y_column,
                        marginal_x="histogram",
                        marginal_y="histogram",
                        title=f"Density Contour: {y_column} vs {x_column}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 3D Scatter
        elif relationship_type == "3D Scatter":
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 3:
                st.warning("3D scatter plot requires at least 3 numeric columns.")
            else:
                # Select x, y, and z columns
                x_column = st.selectbox(
                    "Select x-axis column",
                    options=numeric_columns
                )
                
                y_column = st.selectbox(
                    "Select y-axis column",
                    options=[col for col in numeric_columns if col != x_column],
                    index=0
                )
                
                z_column = st.selectbox(
                    "Select z-axis column",
                    options=[col for col in numeric_columns if col not in [x_column, y_column]],
                    index=0
                )
                
                # Select categorical column for color (optional)
                categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_column = st.selectbox(
                    "Select a categorical column for color (optional)",
                    options=["None"] + categorical_columns
                )
                
                # Create 3D scatter plot
                if color_column != "None":
                    fig = px.scatter_3d(
                        df,
                        x=x_column,
                        y=y_column,
                        z=z_column,
                        color=color_column,
                        title=f"3D Scatter Plot: {x_column}, {y_column}, {z_column}"
                    )
                else:
                    fig = px.scatter_3d(
                        df,
                        x=x_column,
                        y=y_column,
                        z=z_column,
                        title=f"3D Scatter Plot: {x_column}, {y_column}, {z_column}"
                    )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Geospatial
    with chart_tabs[4]:
        st.subheader("Geospatial Visualizations")
        
        st.markdown("""
        Visualize data on maps and explore geographical patterns and distributions.
        """)
        
        # Check if the dataset has location columns
        location_keywords = ['country', 'state', 'city', 'province', 'region', 'county', 'lat', 'lon', 'latitude', 'longitude', 'postal', 'zip']
        
        potential_location_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in location_keywords):
                potential_location_columns.append(col)
        
        if not potential_location_columns:
            st.info("No location columns detected. Please select columns manually if your dataset contains geographic data.")
            
            # Option to create demo geospatial visualization
            if st.checkbox("Create a demo geospatial visualization"):
                create_demo_geospatial()
        
        # Geospatial chart options
        geo_type = st.selectbox(
            "Select geospatial visualization type",
            options=["Choropleth Map", "Scatter Geo", "Bubble Map", "Line Map"]
        )
        
        # Choropleth Map
        if geo_type == "Choropleth Map":
            st.markdown("""
            A choropleth map uses different colors or shading to represent values across geographic regions.
            It requires a column with location identifiers (country/state/region names) and a numeric column for values.
            """)
            
            # Select location column
            location_column = st.selectbox(
                "Select location column",
                options=df.select_dtypes(include=['object', 'category']).columns.tolist(),
                index=0 if potential_location_columns else None
            )
            
            # Select value column
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_columns:
                value_column = st.selectbox(
                    "Select value column",
                    options=numeric_columns
                )
                
                # Select location type
                location_type = st.selectbox(
                    "Select location type",
                    options=["Country", "USA State", "Custom"]
                )
                
                # Aggregation method
                agg_method = st.selectbox(
                    "Select aggregation method",
                    options=["Mean", "Sum", "Count", "Median", "Min", "Max"]
                )
                
                # Map aggregation method to pandas function
                agg_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Count": "count",
                    "Median": "median",
                    "Min": "min",
                    "Max": "max"
                }
                
                # Aggregate data by location
                agg_data = df.groupby(location_column)[value_column].agg(agg_map[agg_method]).reset_index()
                
                # Create choropleth map
                if location_type == "Country":
                    fig = px.choropleth(
                        agg_data,
                        locations=location_column,
                        locationmode="country names",
                        color=value_column,
                        color_continuous_scale="Viridis",
                        title=f"Choropleth Map: {agg_method} of {value_column} by Country",
                        labels={value_column: f"{agg_method} of {value_column}"}
                    )
                elif location_type == "USA State":
                    fig = px.choropleth(
                        agg_data,
                        locations=location_column,
                        locationmode="USA-states",
                        color=value_column,
                        scope="usa",
                        color_continuous_scale="Viridis",
                        title=f"Choropleth Map: {agg_method} of {value_column} by US State",
                        labels={value_column: f"{agg_method} of {value_column}"}
                    )
                else:  # Custom
                    # Try ISO codes first
                    try:
                        fig = px.choropleth(
                            agg_data,
                            locations=location_column,
                            locationmode="ISO-3",
                            color=value_column,
                            color_continuous_scale="Viridis",
                            title=f"Choropleth Map: {agg_method} of {value_column}",
                            labels={value_column: f"{agg_method} of {value_column}"}
                        )
                    except:
                        st.warning("Could not create map with ISO-3 codes. Try using country names or another location type.")
                        # Fall back to country names
                        fig = px.choropleth(
                            agg_data,
                            locations=location_column,
                            locationmode="country names",
                            color=value_column,
                            color_continuous_scale="Viridis",
                            title=f"Choropleth Map: {agg_method} of {value_column}",
                            labels={value_column: f"{agg_method} of {value_column}"}
                        )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns found for value mapping.")
        
        # Scatter Geo and Bubble Map
        elif geo_type in ["Scatter Geo", "Bubble Map"]:
            # Check for latitude and longitude columns
            lat_col = None
            lon_col = None
            
            for col in df.columns:
                if col.lower() in ['lat', 'latitude']:
                    lat_col = col
                elif col.lower() in ['lon', 'long', 'longitude']:
                    lon_col = col
            
            # Select latitude column
            lat_column = st.selectbox(
                "Select latitude column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                index=df.select_dtypes(include=['int64', 'float64']).columns.tolist().index(lat_col) if lat_col else 0
            )
            
            # Select longitude column
            lon_column = st.selectbox(
                "Select longitude column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                index=df.select_dtypes(include=['int64', 'float64']).columns.tolist().index(lon_col) if lon_col else 0
            )
            
            # For bubble map, select size column
            if geo_type == "Bubble Map":
                numeric_columns = [col for col in df.select_dtypes(include=['int64', 'float64']).columns.tolist() 
                                   if col not in [lat_column, lon_column]]
                
                if numeric_columns:
                    size_column = st.selectbox(
                        "Select bubble size column",
                        options=numeric_columns
                    )
                else:
                    size_column = None
                    st.warning("No additional numeric columns found for bubble sizing.")
            
            # Select color column (optional)
            color_options = ["None"] + [col for col in df.columns if col not in [lat_column, lon_column]]
            color_column = st.selectbox(
                "Select color column (optional)",
                options=color_options
            )
            
            # Select map scope
            scope = st.selectbox(
                "Select map scope",
                options=["world", "usa", "europe", "asia", "africa", "north america", "south america"]
            )
            
            # Create map
            if geo_type == "Scatter Geo":
                if color_column != "None":
                    fig = px.scatter_geo(
                        df,
                        lat=lat_column,
                        lon=lon_column,
                        color=color_column,
                        scope=scope,
                        title="Scatter Geo Map"
                    )
                else:
                    fig = px.scatter_geo(
                        df,
                        lat=lat_column,
                        lon=lon_column,
                        scope=scope,
                        title="Scatter Geo Map"
                    )
            else:  # Bubble Map
                if size_column:
                    if color_column != "None":
                        fig = px.scatter_geo(
                            df,
                            lat=lat_column,
                            lon=lon_column,
                            size=size_column,
                            color=color_column,
                            scope=scope,
                            title=f"Bubble Map (sized by {size_column})"
                        )
                    else:
                        fig = px.scatter_geo(
                            df,
                            lat=lat_column,
                            lon=lon_column,
                            size=size_column,
                            scope=scope,
                            title=f"Bubble Map (sized by {size_column})"
                        )
                else:
                    # Fall back to scatter geo
                    if color_column != "None":
                        fig = px.scatter_geo(
                            df,
                            lat=lat_column,
                            lon=lon_column,
                            color=color_column,
                            scope=scope,
                            title="Scatter Geo Map"
                        )
                    else:
                        fig = px.scatter_geo(
                            df,
                            lat=lat_column,
                            lon=lon_column,
                            scope=scope,
                            title="Scatter Geo Map"
                        )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Line Map
        elif geo_type == "Line Map":
            st.markdown("""
            A line map connects points on a map, useful for showing routes, flows, or connections between locations.
            It requires columns for start and end coordinates.
            """)
            
            # Check for latitude and longitude columns
            lat_col = None
            lon_col = None
            
            for col in df.columns:
                if col.lower() in ['lat', 'latitude']:
                    lat_col = col
                elif col.lower() in ['lon', 'long', 'longitude']:
                    lon_col = col
            
            # Ask for source and destination coordinates
            st.subheader("Source Coordinates")
            source_lat = st.selectbox(
                "Select source latitude column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                index=df.select_dtypes(include=['int64', 'float64']).columns.tolist().index(lat_col) if lat_col else 0,
                key="source_lat"
            )
            
            source_lon = st.selectbox(
                "Select source longitude column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                index=df.select_dtypes(include=['int64', 'float64']).columns.tolist().index(lon_col) if lon_col else 0,
                key="source_lon"
            )
            
            st.subheader("Destination Coordinates")
            dest_lat = st.selectbox(
                "Select destination latitude column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                index=df.select_dtypes(include=['int64', 'float64']).columns.tolist().index(lat_col) if lat_col else 0,
                key="dest_lat"
            )
            
            dest_lon = st.selectbox(
                "Select destination longitude column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                index=df.select_dtypes(include=['int64', 'float64']).columns.tolist().index(lon_col) if lon_col else 0,
                key="dest_lon"
            )
            
            # Select map scope
            scope = st.selectbox(
                "Select map scope",
                options=["world", "usa", "europe", "asia", "africa", "north america", "south america"],
                key="line_map_scope"
            )
            
            # Check if necessary columns exist
            if all([source_lat, source_lon, dest_lat, dest_lon]):
                # Select color column (optional)
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                color_column = st.selectbox(
                    "Select color column for lines (optional)",
                    options=["None"] + numeric_columns
                )
                
                # Create line map
                fig = go.Figure()
                
                # Add lines between source and destination
                for _, row in df.iterrows():
                    fig.add_trace(
                        go.Scattergeo(
                            lon=[row[source_lon], row[dest_lon]],
                            lat=[row[source_lat], row[dest_lat]],
                            mode='lines',
                            line=dict(width=1, color='red'),
                            opacity=0.8,
                            showlegend=False
                        )
                    )
                
                # Add source points
                fig.add_trace(
                    go.Scattergeo(
                        lon=df[source_lon],
                        lat=df[source_lat],
                        text=df.index,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='blue',
                            line=dict(width=1, color='blue')
                        ),
                        name='Source'
                    )
                )
                
                # Add destination points
                fig.add_trace(
                    go.Scattergeo(
                        lon=df[dest_lon],
                        lat=df[dest_lat],
                        text=df.index,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='green',
                            line=dict(width=1, color='green')
                        ),
                        name='Destination'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title='Route Map',
                    geo=dict(
                        scope=scope,
                        showland=True,
                        landcolor='rgb(243, 243, 243)',
                        countrycolor='rgb(204, 204, 204)',
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select source and destination coordinates to create a line map.")
    
    # Tab 6: Custom Charts
    with chart_tabs[5]:
        st.subheader("Custom Chart Builder")
        
        st.markdown("""
        Create custom visualizations by directly selecting chart types and configuring parameters.
        """)
        
        # Chart type selection
        chart_type = st.selectbox(
            "Select chart type",
            options=[
                "Bar Chart", "Line Chart", "Area Chart", "Scatter Plot", "Pie Chart", 
                "Box Plot", "Violin Plot", "Histogram", "Heatmap", "3D Surface"
            ]
        )
        
        # Configuration based on chart type
        if chart_type in ["Bar Chart", "Line Chart", "Area Chart", "Scatter Plot"]:
            # These charts need x and y axes
            x_column = st.selectbox(
                "Select x-axis column",
                options=df.columns.tolist()
            )
            
            y_columns = st.multiselect(
                "Select y-axis column(s)",
                options=[col for col in df.columns.tolist() if col != x_column],
                default=[col for col in df.select_dtypes(include=['int64', 'float64']).columns.tolist() 
                         if col != x_column][:1]
            )
            
            if y_columns:
                # Color option
                color_column = st.selectbox(
                    "Select color column (optional)",
                    options=["None"] + [col for col in df.columns.tolist() if col not in [x_column] + y_columns]
                )
                
                # Create chart
                if chart_type == "Bar Chart":
                    # Bar mode for multiple columns
                    if len(y_columns) > 1:
                        bar_mode = st.selectbox(
                            "Bar mode",
                            options=["group", "stack", "relative"]
                        )
                    else:
                        bar_mode = "group"
                    
                    # Orientation
                    orientation = st.selectbox(
                        "Orientation",
                        options=["vertical", "horizontal"]
                    )
                    
                    if orientation == "vertical":
                        if color_column != "None":
                            fig = px.bar(
                                df,
                                x=x_column,
                                y=y_columns,
                                color=color_column,
                                barmode=bar_mode,
                                title=f"Bar Chart: {', '.join(y_columns)} by {x_column}"
                            )
                        else:
                            fig = px.bar(
                                df,
                                x=x_column,
                                y=y_columns,
                                barmode=bar_mode,
                                title=f"Bar Chart: {', '.join(y_columns)} by {x_column}"
                            )
                    else:  # horizontal
                        # Horizontal orientation only works with one y column
                        y_col = y_columns[0]
                        if color_column != "None":
                            fig = px.bar(
                                df,
                                x=y_col,
                                y=x_column,
                                color=color_column,
                                orientation='h',
                                title=f"Bar Chart: {y_col} by {x_column}"
                            )
                        else:
                            fig = px.bar(
                                df,
                                x=y_col,
                                y=x_column,
                                orientation='h',
                                title=f"Bar Chart: {y_col} by {x_column}"
                            )
                
                elif chart_type == "Line Chart":
                    # Line shape
                    line_shape = st.selectbox(
                        "Line shape",
                        options=["linear", "spline", "hv", "vh", "hvh", "vhv"]
                    )
                    
                    # Show markers
                    markers = st.checkbox("Show markers", value=True)
                    
                    if color_column != "None":
                        fig = px.line(
                            df,
                            x=x_column,
                            y=y_columns,
                            color=color_column,
                            markers=markers,
                            line_shape=line_shape,
                            title=f"Line Chart: {', '.join(y_columns)} by {x_column}"
                        )
                    else:
                        fig = px.line(
                            df,
                            x=x_column,
                            y=y_columns,
                            markers=markers,
                            line_shape=line_shape,
                            title=f"Line Chart: {', '.join(y_columns)} by {x_column}"
                        )
                
                elif chart_type == "Area Chart":
                    # Group mode
                    group_mode = st.selectbox(
                        "Group mode",
                        options=["stack", "group", "overlay", "relative"]
                    )
                    
                    if color_column != "None":
                        fig = px.area(
                            df,
                            x=x_column,
                            y=y_columns,
                            color=color_column,
                            groupnorm="percent" if group_mode == "relative" else None,
                            title=f"Area Chart: {', '.join(y_columns)} by {x_column}"
                        )
                    else:
                        fig = px.area(
                            df,
                            x=x_column,
                            y=y_columns,
                            groupnorm="percent" if group_mode == "relative" else None,
                            title=f"Area Chart: {', '.join(y_columns)} by {x_column}"
                        )
                
                elif chart_type == "Scatter Plot":
                    # Trendline
                    trendline = st.checkbox("Show trendline", value=False)
                    
                    # Check if statsmodels is available
                    try:
                        import statsmodels.api as sm
                        has_statsmodels = True
                    except ImportError:
                        has_statsmodels = False
                        if trendline:
                            st.warning("Note: Statsmodels package is required for trendlines. Showing plot without trendline.")
                    
                    # Marker size
                    size_column = st.selectbox(
                        "Select marker size column (optional)",
                        options=["None"] + df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    )
                    
                    if color_column != "None" and size_column != "None":
                        fig = px.scatter(
                            df,
                            x=x_column,
                            y=y_columns[0],  # Scatter plot only supports one y column
                            color=color_column,
                            size=size_column,
                            trendline="ols" if trendline and has_statsmodels else None,
                            title=f"Scatter Plot: {y_columns[0]} vs {x_column}"
                        )
                    elif color_column != "None":
                        fig = px.scatter(
                            df,
                            x=x_column,
                            y=y_columns[0],
                            color=color_column,
                            trendline="ols" if trendline and has_statsmodels else None,
                            title=f"Scatter Plot: {y_columns[0]} vs {x_column}"
                        )
                    elif size_column != "None":
                        fig = px.scatter(
                            df,
                            x=x_column,
                            y=y_columns[0],
                            size=size_column,
                            trendline="ols" if trendline and has_statsmodels else None,
                            title=f"Scatter Plot: {y_columns[0]} vs {x_column}"
                        )
                    else:
                        fig = px.scatter(
                            df,
                            x=x_column,
                            y=y_columns[0],
                            trendline="ols" if trendline and has_statsmodels else None,
                            title=f"Scatter Plot: {y_columns[0]} vs {x_column}"
                        )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one y-axis column.")
        
        elif chart_type == "Pie Chart":
            # Pie chart needs name and value columns
            name_column = st.selectbox(
                "Select name column (categories)",
                options=df.select_dtypes(include=['object', 'category']).columns.tolist()
            )
            
            value_column = st.selectbox(
                "Select value column",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            )
            
            # Hole parameter (donut chart)
            hole = st.slider("Hole size (0 for pie chart, >0 for donut chart)", 
                            min_value=0.0, max_value=0.8, value=0.0, step=0.1)
            
            # Create pie/donut chart
            fig = px.pie(
                df,
                names=name_column,
                values=value_column,
                title=f"Pie Chart: {value_column} by {name_column}",
                hole=hole
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type in ["Box Plot", "Violin Plot"]:
            # These charts need y axis and optional x axis
            y_column = st.selectbox(
                "Select y-axis column (values)",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            )
            
            x_column = st.selectbox(
                "Select x-axis column (categories, optional)",
                options=["None"] + df.select_dtypes(include=['object', 'category']).columns.tolist()
            )
            
            # Color option
            color_column = st.selectbox(
                "Select color column (optional)",
                options=["None"] + [col for col in df.columns.tolist() if col not in [x_column, y_column]]
            )
            
            # Create chart
            if chart_type == "Box Plot":
                if x_column != "None" and color_column != "None":
                    fig = px.box(
                        df,
                        x=x_column,
                        y=y_column,
                        color=color_column,
                        points="all",
                        title=f"Box Plot of {y_column}"
                    )
                elif x_column != "None":
                    fig = px.box(
                        df,
                        x=x_column,
                        y=y_column,
                        points="all",
                        title=f"Box Plot of {y_column} by {x_column}"
                    )
                elif color_column != "None":
                    fig = px.box(
                        df,
                        y=y_column,
                        color=color_column,
                        points="all",
                        title=f"Box Plot of {y_column}"
                    )
                else:
                    fig = px.box(
                        df,
                        y=y_column,
                        points="all",
                        title=f"Box Plot of {y_column}"
                    )
            
            elif chart_type == "Violin Plot":
                # Include box plot inside violin
                box = st.checkbox("Include box plot", value=True)
                
                if x_column != "None" and color_column != "None":
                    fig = px.violin(
                        df,
                        x=x_column,
                        y=y_column,
                        color=color_column,
                        box=box,
                        points="all",
                        title=f"Violin Plot of {y_column}"
                    )
                elif x_column != "None":
                    fig = px.violin(
                        df,
                        x=x_column,
                        y=y_column,
                        box=box,
                        points="all",
                        title=f"Violin Plot of {y_column} by {x_column}"
                    )
                elif color_column != "None":
                    fig = px.violin(
                        df,
                        y=y_column,
                        color=color_column,
                        box=box,
                        points="all",
                        title=f"Violin Plot of {y_column}"
                    )
                else:
                    fig = px.violin(
                        df,
                        y=y_column,
                        box=box,
                        points="all",
                        title=f"Violin Plot of {y_column}"
                    )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Histogram":
            # Histogram needs a value column
            column = st.selectbox(
                "Select column for histogram",
                options=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            )
            
            # Bins parameter
            bins = st.slider("Number of bins", min_value=5, max_value=100, value=20)
            
            # Color option
            color_column = st.selectbox(
                "Select color column (optional)",
                options=["None"] + [col for col in df.columns.tolist() if col != column]
            )
            
            # Marginal plot
            marginal = st.selectbox(
                "Marginal plot",
                options=["None", "box", "violin", "rug"]
            )
            
            # Create histogram
            if color_column != "None":
                fig = px.histogram(
                    df,
                    x=column,
                    color=color_column,
                    nbins=bins,
                    marginal=None if marginal == "None" else marginal,
                    title=f"Histogram of {column}"
                )
            else:
                fig = px.histogram(
                    df,
                    x=column,
                    nbins=bins,
                    marginal=None if marginal == "None" else marginal,
                    title=f"Histogram of {column}"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Heatmap":
            # Heatmap needs two categorical columns and one numeric column
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(categorical_columns) < 2 or not numeric_columns:
                st.warning("Heatmap requires at least 2 categorical columns and 1 numeric column.")
            else:
                # Select x and y categorical columns
                x_column = st.selectbox(
                    "Select x-axis column (categorical)",
                    options=categorical_columns,
                    key="heatmap_x"
                )
                
                y_column = st.selectbox(
                    "Select y-axis column (categorical)",
                    options=[col for col in categorical_columns if col != x_column],
                    index=0,
                    key="heatmap_y"
                )
                
                # Select numeric column for values
                value_column = st.selectbox(
                    "Select value column (numeric)",
                    options=numeric_columns,
                    key="heatmap_value"
                )
                
                # Select aggregation method
                agg_method = st.selectbox(
                    "Select aggregation method",
                    options=["Mean", "Sum", "Count", "Median", "Min", "Max"],
                    key="heatmap_agg"
                )
                
                # Map aggregation method to pandas function
                agg_map = {
                    "Mean": "mean",
                    "Sum": "sum",
                    "Count": "count",
                    "Median": "median",
                    "Min": "min",
                    "Max": "max"
                }
                
                # Create pivot table for heatmap
                pivot_df = df.pivot_table(
                    values=value_column,
                    index=y_column,
                    columns=x_column,
                    aggfunc=agg_map[agg_method]
                )
                
                # Create heatmap
                fig = px.imshow(
                    pivot_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=st.selectbox(
                        "Color scale",
                        options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", 
                                "Blues", "Greens", "Reds", "YlOrRd", "RdBu", "RdBu_r"]
                    ),
                    title=f"Heatmap: {agg_method} of {value_column} by {x_column} and {y_column}"
                )
                
                fig.update_layout(
                    xaxis_title=x_column,
                    yaxis_title=y_column
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "3D Surface":
            # 3D surface needs three numeric columns
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numeric_columns) < 3:
                st.warning("3D surface plot requires at least 3 numeric columns.")
            else:
                # Select x, y, and z columns
                x_column = st.selectbox(
                    "Select x-axis column",
                    options=numeric_columns,
                    key="surface_x"
                )
                
                y_column = st.selectbox(
                    "Select y-axis column",
                    options=[col for col in numeric_columns if col != x_column],
                    index=0,
                    key="surface_y"
                )
                
                z_column = st.selectbox(
                    "Select z-axis column (values)",
                    options=[col for col in numeric_columns if col not in [x_column, y_column]],
                    index=0,
                    key="surface_z"
                )
                
                # Create 3D surface plot
                fig = go.Figure(data=[go.Surface(
                    z=df.pivot_table(
                        values=z_column,
                        index=y_column,
                        columns=x_column,
                        aggfunc='mean'
                    ).values,
                    x=df[x_column].unique(),
                    y=df[y_column].unique(),
                    colorscale=st.selectbox(
                        "Color scale",
                        options=["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo", 
                                "Blues", "Greens", "Reds", "YlOrRd", "RdBu", "RdBu_r"],
                        key="surface_color"
                    )
                )])
                
                fig.update_layout(
                    title=f"3D Surface Plot: {z_column} by {x_column} and {y_column}",
                    scene=dict(
                        xaxis_title=x_column,
                        yaxis_title=y_column,
                        zaxis_title=z_column
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

def create_demo_timeseries():
    """
    Create a demo time series visualization with random data.
    """
    st.subheader("Demo Time Series Visualization")
    
    # Generate demo time series data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 1, 1)
    
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]
    
    # Generate random values
    np.random.seed(42)  # For reproducibility
    values1 = np.cumsum(np.random.normal(0, 1, len(dates)))
    values2 = np.cumsum(np.random.normal(0, 1, len(dates))) + 10
    
    # Create DataFrame
    demo_df = pd.DataFrame({
        'Date': dates,
        'Metric A': values1,
        'Metric B': values2
    })
    
    # Display sample data
    st.markdown("### Sample Time Series Data")
    st.dataframe(demo_df.head())
    
    # Create a line chart
    fig = px.line(
        demo_df,
        x='Date',
        y=['Metric A', 'Metric B'],
        title='Demo Time Series: Metrics Over Time',
        markers=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create area chart
    fig = px.area(
        demo_df,
        x='Date',
        y=['Metric A', 'Metric B'],
        title='Demo Area Chart: Metrics Over Time'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly aggregation
    demo_df['Month'] = pd.to_datetime(demo_df['Date']).dt.to_period('M').astype(str)
    monthly_data = demo_df.groupby('Month')[['Metric A', 'Metric B']].mean().reset_index()
    
    # Create bar chart for monthly data
    fig = px.bar(
        monthly_data,
        x='Month',
        y=['Metric A', 'Metric B'],
        title='Demo Monthly Aggregation',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_demo_geospatial():
    """
    Create a demo geospatial visualization with sample data.
    """
    st.subheader("Demo Geospatial Visualization")
    
    # Sample data for countries
    countries = ['United States', 'Canada', 'Mexico', 'Brazil', 'United Kingdom', 
                'France', 'Germany', 'Italy', 'China', 'Japan', 'Australia', 'India']
    
    # Generate random values
    np.random.seed(42)  # For reproducibility
    values = np.random.randint(10, 100, len(countries))
    
    # Create DataFrame
    demo_df = pd.DataFrame({
        'Country': countries,
        'Value': values
    })
    
    # Display sample data
    st.markdown("### Sample Geospatial Data")
    st.dataframe(demo_df)
    
    # Create choropleth map
    fig = px.choropleth(
        demo_df,
        locations='Country',
        locationmode='country names',
        color='Value',
        color_continuous_scale='Viridis',
        title='Demo Choropleth Map: Values by Country'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample data for cities
    cities = [
        'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
        'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'
    ]
    
    # Approximate coordinates for these cities
    latitudes = [
        40.7128, 34.0522, 41.8781, 29.7604, 33.4484,
        39.9526, 29.4241, 32.7157, 32.7767, 37.3382
    ]
    
    longitudes = [
        -74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
        -75.1652, -98.4936, -117.1611, -96.7970, -121.8863
    ]
    
    # Generate random values
    values = np.random.randint(20, 200, len(cities))
    
    # Create DataFrame
    city_df = pd.DataFrame({
        'City': cities,
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Value': values
    })
    
    # Display sample data
    st.markdown("### Sample City Data")
    st.dataframe(city_df)
    
    # Create bubble map
    fig = px.scatter_geo(
        city_df,
        lat='Latitude',
        lon='Longitude',
        text='City',
        size='Value',
        color='Value',
        color_continuous_scale='Viridis',
        title='Demo Bubble Map: US Cities',
        scope='usa'
    )
    
    st.plotly_chart(fig, use_container_width=True)