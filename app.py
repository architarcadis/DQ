import streamlit as st
import pandas as pd
from components.sidebar import render_sidebar
import sys
import os

# Set path to allow importing modules from the modules directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

# Set page configuration
st.set_page_config(
    page_title="Arcadis Data Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = None

if 'llm_config' not in st.session_state:
    st.session_state.llm_config = {
        "type": "OpenAI API",
        "key": "",
        "model_path": ""
    }

if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

# Custom CSS to hide Streamlit's default navigation and simplify UI
hide_streamlit_style = """
<style>
    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Reduce padding at the top */
    div.block-container {padding-top: 1rem;}
    
    /* Hide default Streamlit navigation elements */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Hide default sidebar navigation elements */
    section[data-testid="stSidebar"] div.sidebar-content {
        margin-top: -80px;
    }
    
    /* Additional selectors to hide sidebar navigation items */
    .sidebar .sidebar-content {
        background-color: white;
    }
    .sidebar .sidebar-content .sidebar-section > div:first-child {
        display: none !important;
    }
    .sidebar-content > div:first-child {
        margin-top: -50px;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Render sidebar (with only upload and API settings)
render_sidebar()

# Main navigation tabs at the top
st.title("🔍 Arcadis Data Analytics Platform")

# Define tab names
tab_names = ["Home", "Data Hub", "Data Quality Assessment", "Robust Charts", "Advanced Analytics", "Ask the Data"]

# Create tabs
tabs = st.tabs(tab_names)

# Render content based on selected tab
# Tab 0: Home
with tabs[0]:
    st.markdown("""
    Welcome to the Arcadis Data Analytics Platform, a comprehensive tool for exploring, visualizing, and analyzing data.

    ### Key Features
    - **📈 Data Hub**: Combine and transform datasets
    - **🔍 Data Quality Assessment**: Comprehensive data quality analysis
    - **📊 Robust Charts**: Pre-populated advanced visualizations
    - **🤖 Advanced Analytics**: Leverage machine learning models
    - **💬 Ask the Data**: Query your data in natural language

    ### Getting Started
    1. Upload data files using the sidebar on the left
    2. Navigate to different tools using the tabs above
    3. Create visualizations and gain insights from your data
    """)

    # Display sample visualizations if no data is loaded
    if not st.session_state.datasets:
        st.info("👈 Start by uploading some data files using the sidebar on the left", icon="ℹ️")
        
        # Sample dashboard preview
        st.subheader("Sample Dashboard Preview")
        
        cols = st.columns(3)
        with cols[0]:
            st.metric(label="Total Sales", value="$1.2M", delta="8.5%")
        
        with cols[1]:
            st.metric(label="Customer Count", value="12,390", delta="-2.1%")
        
        with cols[2]:
            st.metric(label="Average Order", value="$97.30", delta="4.3%")
        
        # Add some additional info about the platform
        st.markdown("---")
        st.subheader("About this Platform")
        st.markdown("""
        The Arcadis Data Analytics Platform is designed to:
        - Streamline data analysis workflows
        - Create compelling data visualizations
        - Enable non-technical users to derive insights
        - Support data-driven decision making
        
        Click on the tabs above to explore different components of the platform.
        """)
    else:
        # Show active dataset if available
        st.subheader(f"Active Dataset: {st.session_state.active_dataset}")
        active_df = st.session_state.datasets[st.session_state.active_dataset]
        
        # Display dataset preview
        st.dataframe(active_df.head(5), use_container_width=True)
        
        # Show basic stats
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", active_df.shape[0])
        with col2:
            st.metric("Columns", active_df.shape[1])
        with col3:
            memory_usage = active_df.memory_usage(deep=True).sum()
            if memory_usage > 1024 * 1024:
                memory_str = f"{memory_usage / (1024 * 1024):.2f} MB"
            else:
                memory_str = f"{memory_usage / 1024:.2f} KB"
            st.metric("Memory Usage", memory_str)

# Tab 1: Data Hub
with tabs[1]:
    # Import from the modules directory
    from data_hub import show_data_hub
    show_data_hub()

# Tab 2: Data Quality Assessment
with tabs[2]:
    from data_quality_assessment import show_data_quality_assessment
    show_data_quality_assessment()

# Tab 3: Robust Charts
with tabs[3]:
    from robust_charts import show_robust_charts
    show_robust_charts()

# Tab 4: Advanced Analytics
with tabs[4]:
    from advanced_analytics import show_advanced_analytics
    show_advanced_analytics()

# Tab 5: Ask the Data
with tabs[5]:
    from ask_the_data import show_ask_the_data
    show_ask_the_data()