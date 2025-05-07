import streamlit as st
import plotly.express as px
import pandas as pd
import uuid
from utils.viz_helpers import create_donut_chart, create_heatmap, create_gantt, arc_gauge
from utils.llm import analysis_chain

def show_executive_summary():
    # Main content
    st.subheader("📊 Executive Summary Builder")
    st.markdown("Create interactive dashboards with drag-and-drop visualization components")

    # Check if datasets exist
    if not st.session_state.datasets:
        st.info("👆 Start by uploading some data files using the sidebar", icon="ℹ️")
        
        # Using a stock photo for visual appeal
        st.image("https://pixabay.com/get/g1ddab242cef5a072c1c1646c72be328baa64b48cb9e8d2ec4e12e285c6ac76e0f27b902eec5067e98367eafd935a9921c505b716ee8e7666c6929e1386046c09_1280.jpg", 
                caption="Build informative executive dashboards", 
                width=500)
        return

    # Initialize dashboard elements if not exists
    if 'dashboard_elements' not in st.session_state:
        st.session_state.dashboard_elements = []

    # Component palette and dashboard canvas
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Component Palette")
        
        # Component selection
        component_type = st.selectbox(
            "Select a component type:",
            options=["KPI Card", "Donut Chart", "Speedometer", "Heatmap", "Gantt Chart", "Bar Chart", "Line Chart"],
            index=0
        )
        
        # Dataset selection
        if st.session_state.datasets:
            dataset_name = st.selectbox(
                "Dataset:",
                options=list(st.session_state.datasets.keys()),
                index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) 
                    if st.session_state.active_dataset in st.session_state.datasets 
                    else 0
            )
            
            df = st.session_state.datasets[dataset_name]
            
            # Component configuration based on type
            with st.form(key="component_config"):
                st.subheader("Component Settings")
                
                # Common settings
                title = st.text_input("Title:", f"New {component_type}")
                
                # Component-specific settings
                if component_type == "KPI Card":
                    metric_column = st.selectbox("Metric column:", options=df.select_dtypes(include=['number']).columns)
                    aggregation = st.selectbox("Aggregation:", options=["Sum", "Average", "Min", "Max", "Count"])
                    prefix = st.text_input("Prefix:", value="")
                    suffix = st.text_input("Suffix:", value="")
                    
                elif component_type == "Donut Chart":
                    value_column = st.selectbox("Value column:", options=df.select_dtypes(include=['number']).columns)
                    category_column = st.selectbox("Category column:", options=df.select_dtypes(exclude=['number']).columns)
                    
                elif component_type == "Speedometer":
                    value_column = st.selectbox("Value column:", options=df.select_dtypes(include=['number']).columns)
                    aggregation = st.selectbox("Aggregation:", options=["Sum", "Average", "Min", "Max"])
                    min_value = st.number_input("Minimum value:", value=0.0)
                    max_value = st.number_input("Maximum value:", value=100.0)
                    
                elif component_type == "Heatmap":
                    x_column = st.selectbox("X-axis column:", options=df.columns)
                    y_column = st.selectbox("Y-axis column:", options=[c for c in df.columns if c != x_column])
                    value_column = st.selectbox(
                        "Value column:", 
                        options=[c for c in df.select_dtypes(include=['number']).columns if c != x_column and c != y_column]
                    )
                    
                elif component_type == "Gantt Chart":
                    task_column = st.selectbox("Task column:", options=df.select_dtypes(exclude=['number']).columns)
                    # Get datetime columns, fallback to all if none are found
                    date_columns = df.select_dtypes(include=['datetime', 'date']).columns
                    if len(date_columns) == 0:
                        date_columns = df.columns
                    
                    start_column = st.selectbox("Start date column:", options=date_columns)
                    
                    # Filter for end date column
                    remaining_date_cols = [c for c in date_columns if c != start_column]
                    end_column = st.selectbox(
                        "End date column:", 
                        options=remaining_date_cols if remaining_date_cols else date_columns
                    )
                    
                    category_column = st.selectbox(
                        "Category column:", 
                        options=[c for c in df.select_dtypes(exclude=['number']).columns if c != task_column]
                    )
                    
                elif component_type == "Bar Chart":
                    x_column = st.selectbox("X-axis column:", options=df.columns)
                    y_column = st.selectbox(
                        "Y-axis column:", 
                        options=[c for c in df.select_dtypes(include=['number']).columns if c != x_column]
                    )
                    color_column = st.selectbox(
                        "Color by (optional):", 
                        options=["None"] + [c for c in df.columns if c != x_column and c != y_column]
                    )
                    
                elif component_type == "Line Chart":
                    x_column = st.selectbox("X-axis column:", options=df.columns)
                    y_columns = st.multiselect(
                        "Y-axis column(s):", 
                        options=[c for c in df.select_dtypes(include=['number']).columns if c != x_column]
                    )
                
                # Add component button
                submit_button = st.form_submit_button("Add Component", use_container_width=True)
                
                if submit_button:
                    # Create a new component configuration
                    component_id = str(uuid.uuid4())
                    component_config = {
                        "id": component_id,
                        "type": component_type,
                        "title": title,
                        "dataset": dataset_name,
                        "config": {}
                    }
                    
                    # Add component-specific configuration
                    if component_type == "KPI Card":
                        component_config["config"] = {
                            "metric_column": metric_column,
                            "aggregation": aggregation,
                            "prefix": prefix,
                            "suffix": suffix
                        }
                        
                    elif component_type == "Donut Chart":
                        component_config["config"] = {
                            "value_column": value_column,
                            "category_column": category_column
                        }
                        
                    elif component_type == "Speedometer":
                        component_config["config"] = {
                            "value_column": value_column,
                            "aggregation": aggregation,
                            "min_value": min_value,
                            "max_value": max_value
                        }
                        
                    elif component_type == "Heatmap":
                        component_config["config"] = {
                            "x_column": x_column,
                            "y_column": y_column,
                            "value_column": value_column
                        }
                        
                    elif component_type == "Gantt Chart":
                        component_config["config"] = {
                            "task_column": task_column,
                            "start_column": start_column,
                            "end_column": end_column,
                            "category_column": category_column
                        }
                        
                    elif component_type == "Bar Chart":
                        component_config["config"] = {
                            "x_column": x_column,
                            "y_column": y_column,
                            "color_column": None if color_column == "None" else color_column
                        }
                        
                    elif component_type == "Line Chart":
                        component_config["config"] = {
                            "x_column": x_column,
                            "y_columns": y_columns
                        }
                    
                    # Add component to dashboard
                    st.session_state.dashboard_elements.append(component_config)
                    st.rerun()

        # AI Insight Button (when dashboard has elements)
        if st.session_state.dashboard_elements:
            if st.button("🤖 Generate AI Insights", use_container_width=True):
                with st.spinner("Analyzing data and generating insights..."):
                    try:
                        # Get the active dataset
                        if st.session_state.active_dataset:
                            df = st.session_state.datasets[st.session_state.active_dataset]
                            
                            # Prepare data summary for LLM
                            data_summary = {
                                "columns": list(df.columns),
                                "stats": df.describe().to_dict(),
                                "dtypes": df.dtypes.astype(str).to_dict(),
                                "sample_size": len(df)
                            }
                            
                            # Call LLM analysis chain
                            insights = analysis_chain(data_summary)
                            
                            # Store insights in session state
                            st.session_state.ai_insights = insights
                            
                            st.success("Successfully generated insights!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")

    # Dashboard canvas
    with col2:
        st.subheader("Dashboard Canvas")
        
        if not st.session_state.dashboard_elements:
            st.info("👈 Add components from the palette to build your dashboard")
        else:
            # Display AI Insights if available
            if st.session_state.ai_insights:
                with st.expander("🤖 AI Insights", expanded=True):
                    for i, insight in enumerate(st.session_state.ai_insights):
                        st.markdown(f"- {insight}")
            
            # Render dashboard elements in a grid
            for i in range(0, len(st.session_state.dashboard_elements), 2):
                cols = st.columns(2)
                
                for j in range(2):
                    if i + j < len(st.session_state.dashboard_elements):
                        element = st.session_state.dashboard_elements[i + j]
                        
                        with cols[j]:
                            with st.container(border=True):
                                # Element header with title and remove button
                                st.subheader(element["title"])
                                
                                # Add remove button at the top right
                                if st.button("🗑️", key=f"remove_{element['id']}", help="Remove this component"):
                                    st.session_state.dashboard_elements.remove(element)
                                    st.rerun()
                                
                                # Render component based on type
                                dataset = st.session_state.datasets[element["dataset"]]
                                
                                try:
                                    if element["type"] == "KPI Card":
                                        config = element["config"]
                                        metric_col = config["metric_column"]
                                        
                                        # Calculate metric based on aggregation
                                        if config["aggregation"] == "Sum":
                                            value = dataset[metric_col].sum()
                                        elif config["aggregation"] == "Average":
                                            value = dataset[metric_col].mean()
                                        elif config["aggregation"] == "Min":
                                            value = dataset[metric_col].min()
                                        elif config["aggregation"] == "Max":
                                            value = dataset[metric_col].max()
                                        elif config["aggregation"] == "Count":
                                            value = dataset[metric_col].count()
                                            
                                        # Format value
                                        formatted_value = f"{config['prefix']}{value:.2f}{config['suffix']}"
                                        
                                        # Display KPI card
                                        st.metric(
                                            label=metric_col,
                                            value=formatted_value
                                        )
                                        
                                    elif element["type"] == "Donut Chart":
                                        config = element["config"]
                                        fig = create_donut_chart(
                                            df=dataset,
                                            value_col=config["value_column"],
                                            name_col=config["category_column"],
                                            title=element["title"]
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    elif element["type"] == "Speedometer":
                                        config = element["config"]
                                        
                                        # Calculate value based on aggregation
                                        if config["aggregation"] == "Sum":
                                            value = dataset[config["value_column"]].sum()
                                        elif config["aggregation"] == "Average":
                                            value = dataset[config["value_column"]].mean()
                                        elif config["aggregation"] == "Min":
                                            value = dataset[config["value_column"]].min()
                                        elif config["aggregation"] == "Max":
                                            value = dataset[config["value_column"]].max()
                                            
                                        # Display speedometer
                                        fig = arc_gauge(
                                            value=value,
                                            min_val=config["min_value"],
                                            max_val=config["max_value"],
                                            title=element["title"]
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    elif element["type"] == "Heatmap":
                                        config = element["config"]
                                        fig = create_heatmap(
                                            df=dataset,
                                            x_col=config["x_column"],
                                            y_col=config["y_column"],
                                            value_col=config["value_column"],
                                            title=element["title"]
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    elif element["type"] == "Gantt Chart":
                                        config = element["config"]
                                        try:
                                            fig = create_gantt(
                                                df=dataset,
                                                task_col=config["task_column"],
                                                start_col=config["start_column"],
                                                end_col=config["end_column"],
                                                color_col=config["category_column"],
                                                title=element["title"]
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error creating Gantt chart: {str(e)}")
                                            st.info("Gantt charts require properly formatted date columns.")
                                        
                                    elif element["type"] == "Bar Chart":
                                        config = element["config"]
                                        # Create bar chart
                                        if config["color_column"]:
                                            fig = px.bar(
                                                dataset,
                                                x=config["x_column"],
                                                y=config["y_column"],
                                                color=config["color_column"],
                                                title=element["title"],
                                                template="arcadis"
                                            )
                                        else:
                                            fig = px.bar(
                                                dataset,
                                                x=config["x_column"],
                                                y=config["y_column"],
                                                title=element["title"],
                                                template="arcadis"
                                            )
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                    elif element["type"] == "Line Chart":
                                        config = element["config"]
                                        # Check if y_columns is a list and not empty
                                        if config["y_columns"] and len(config["y_columns"]) > 0:
                                            # Create line chart
                                            fig = px.line(
                                                dataset,
                                                x=config["x_column"],
                                                y=config["y_columns"],
                                                title=element["title"],
                                                template="arcadis"
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.error("Please select at least one Y-axis column")
                                except Exception as e:
                                    st.error(f"Error rendering component: {str(e)}")
            
            # Export dashboard button
            if st.button("Export Dashboard", use_container_width=True):
                # Create a dictionary representation of the dashboard
                dashboard_export = {
                    "title": "Arcadis Executive Dashboard",
                    "elements": st.session_state.dashboard_elements,
                    "insights": st.session_state.ai_insights
                }
                
                # Convert to JSON string
                dashboard_json = pd.DataFrame([dashboard_export]).to_json(orient="records")
                
                # Offer download
                st.download_button(
                    label="Download Dashboard JSON",
                    data=dashboard_json,
                    file_name="arcadis_dashboard.json",
                    mime="application/json"
                )