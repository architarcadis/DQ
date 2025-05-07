import streamlit as st
import pandas as pd
from utils.extract_data import extract_data

def render_sidebar():
    """
    Renders the sidebar with file upload functionality and LLM settings only.
    """
    with st.sidebar:
        st.title("🛠️ Data Tools")
        
        # File uploader section
        st.subheader("📤 Upload Data")
        
        uploaded_files = st.file_uploader(
            "Upload data files",
            type=["xlsx", "xls", "csv", "pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Supported formats: Excel, CSV, PDF tables, and images"
        )
        
        # Extract data button
        if uploaded_files:
            if st.button("🔄 Load / Extract Data", use_container_width=True):
                with st.spinner("Extracting data from files..."):
                    try:
                        # Call the extract_data function to process uploaded files
                        new_datasets = extract_data(uploaded_files)
                        
                        # Update session state with new datasets
                        st.session_state.datasets.update(new_datasets)
                        
                        # If this is the first dataset, set it as active
                        if new_datasets and not st.session_state.active_dataset:
                            st.session_state.active_dataset = list(new_datasets.keys())[0]
                            
                        st.success(f"Successfully loaded {len(new_datasets)} dataset(s)!")
                    except Exception as e:
                        st.error(f"Error extracting data: {str(e)}")
        
        # LLM settings expander
        with st.expander("🤖 LLM Settings"):
            llm_type = st.radio(
                "LLM Provider:",
                options=["OpenAI API", "Local Model"],
                index=0 if st.session_state.llm_config["type"] == "OpenAI API" else 1
            )
            
            if llm_type == "OpenAI API":
                api_key = st.text_input(
                    "OpenAI API Key:",
                    type="password",
                    value=st.session_state.llm_config.get("key", ""),
                    help="Your OpenAI API key for GPT models"
                )
                
                # Update session state
                st.session_state.llm_config.update({
                    "type": "OpenAI API",
                    "key": api_key
                })
            else:
                model_path = st.text_input(
                    "Local Model Path:",
                    value=st.session_state.llm_config.get("model_path", ""),
                    help="Path to your local LLM model files"
                )
                
                # Update session state
                st.session_state.llm_config.update({
                    "type": "Local Model",
                    "model_path": model_path
                })
        
        # Show current active dataset (if any)
        if st.session_state.datasets and st.session_state.active_dataset:
            st.markdown("---")
            st.caption(f"**Current dataset:** {st.session_state.active_dataset}")
            current_df = st.session_state.datasets[st.session_state.active_dataset]
            st.caption(f"Rows: {current_df.shape[0]} | Columns: {current_df.shape[1]}")
        
        # Add footer with version info
        st.sidebar.markdown("---")
        st.sidebar.caption("Arcadis Data Analytics Platform v1.0")