import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

# Define Arcadis colors
ARCADIS_COLORS = ["#F36F21", "#005EB8", "#78BE20", "#333333", "#9B9B9B"]

# Create Arcadis template for Plotly
pio.templates["arcadis"] = pio.templates["plotly_white"].update(
    layout_font=dict(family="Montserrat,Helvetica Neue,Arial,sans-serif", size=13, color="#333"),
    layout_colorway=ARCADIS_COLORS,
    layout_title_font=dict(size=18, family="Montserrat", color="#333"),
    layout_legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    layout_margin=dict(t=60, r=30, b=80, l=30),
    layout_coloraxis_colorbar=dict(tickfont=dict(family="Montserrat", size=10)),
    layout_xaxis=dict(
        title_font=dict(size=13),
        tickfont=dict(size=11),
        gridcolor="#E6E6E6"
    ),
    layout_yaxis=dict(
        title_font=dict(size=13),
        tickfont=dict(size=11),
        gridcolor="#E6E6E6"
    )
)

# Set as default template
pio.templates.default = "arcadis"

def create_donut_chart(df, value_col, name_col, title):
    """
    Create a donut chart visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    value_col : str
        Column name for the values
    name_col : str
        Column name for the categories
    title : str
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The donut chart figure
    """
    # Aggregate data by category
    grouped_df = df.groupby(name_col)[value_col].sum().reset_index()
    
    # Create donut chart
    fig = px.pie(
        grouped_df,
        values=value_col,
        names=name_col,
        title=title,
        hole=0.6,  # Make it a donut chart
        template="arcadis"
    )
    
    # Update layout
    fig.update_traces(
        textposition='outside',
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_heatmap(df, x_col, y_col, value_col, title):
    """
    Create a heatmap visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    x_col : str
        Column for x-axis categories
    y_col : str
        Column for y-axis categories
    value_col : str
        Column for heat values
    title : str
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The heatmap figure
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values=value_col,
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x=x_col, y=y_col, color=value_col),
        title=title,
        color_continuous_scale=["#F5F5F5", "#F36F21"],
        template="arcadis"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def create_gantt(df, task_col, start_col, end_col, color_col, title):
    """
    Create a Gantt chart visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    task_col : str
        Column for task names
    start_col : str
        Column for start dates
    end_col : str
        Column for end dates
    color_col : str
        Column for color grouping
    title : str
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The Gantt chart figure
    """
    # Ensure date columns are datetime
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])
    
    # Create Gantt chart
    fig = px.timeline(
        df,
        x_start=start_col,
        x_end=end_col,
        y=task_col,
        color=color_col,
        title=title,
        template="arcadis"
    )
    
    # Update layout
    fig.update_yaxes(autorange="reversed")
    
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="Task",
        legend_title=color_col
    )
    
    return fig

def arc_gauge(value, min_val, max_val, title):
    """
    Create a speedometer/gauge visualization.
    
    Parameters:
    -----------
    value : float
        The value to display
    min_val : float
        Minimum value on the gauge
    max_val : float
        Maximum value on the gauge
    title : str
        Chart title
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The gauge figure
    """
    # Calculate percentage for gauge
    percent = (value - min_val) / (max_val - min_val) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add gauge trace
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'family': 'Montserrat', 'color': '#333333'}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#333333"},
            'bar': {'color': "#005EB8"},  # Needle color - Confidence Blue
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#333333",
            'steps': [
                {'range': [min_val, max_val], 'color': '#F5F5F5'},  # Track - Light Grey
                {'range': [min_val, value], 'color': '#F36F21'}     # Arc fill - Heritage Orange
            ],
            'threshold': {
                'line': {'color': "#005EB8", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        template="arcadis",
        margin=dict(l=30, r=30, t=70, b=30),
        height=300
    )
    
    return fig