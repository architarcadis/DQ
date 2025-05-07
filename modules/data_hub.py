import streamlit as st
import pandas as pd
from utils.suggest_keys import suggest_keys
import json

def show_data_hub():
    # Main content
    st.subheader("🔗 Data Hub")
    st.markdown("Explore, transform, and combine your datasets for analysis")

    # Check if datasets exist
    if not st.session_state.datasets:
        st.info("👆 Start by uploading some data files using the sidebar", icon="ℹ️")
        
        # Using a stock photo for visual appeal
        st.image("https://pixabay.com/get/gc4e5e5884c4a159cb6f9433a81372e404c6ae09bff80b78a53e6dd8f4c67fb993f8ce8d6c690b5ccbda2c6a3e781e3d07f911fbd26c116b63a1919df31648f44_1280.jpg", 
                caption="Upload and manage your datasets", 
                width=500)
        return

    # Dataset overview section
    st.subheader("📑 Dataset Overview")

    # Create dataset cards
    cols = st.columns(3)
    for i, (name, df) in enumerate(st.session_state.datasets.items()):
        with cols[i % 3]:
            with st.container(border=True):
                st.subheader(name)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                
                # Show data types and memory usage
                mem_usage = df.memory_usage(deep=True).sum()
                if mem_usage > 1024 * 1024:
                    mem_str = f"{mem_usage / (1024 * 1024):.2f} MB"
                else:
                    mem_str = f"{mem_usage / 1024:.2f} KB"
                
                st.caption(f"Memory: {mem_str}")
                
                # Button to view sample
                if st.button(f"View Sample 👁️", key=f"view_{name}"):
                    st.session_state.active_dataset = name
                    st.rerun()

    # Dataset combining section
    st.markdown("---")
    st.subheader("🔄 Combine Datasets")

    # Dataset selection for joining
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Primary Dataset")
        primary_dataset = st.selectbox(
            "Select primary dataset:",
            options=list(st.session_state.datasets.keys()),
            key="primary_dataset"
        )

    with col2:
        st.markdown("#### Secondary Dataset")
        secondary_dataset = st.selectbox(
            "Select secondary dataset:",
            options=[ds for ds in st.session_state.datasets.keys() if ds != primary_dataset],
            key="secondary_dataset"
        )

    # Get suggested keys if both datasets are selected
    if primary_dataset and secondary_dataset:
        primary_df = st.session_state.datasets[primary_dataset]
        secondary_df = st.session_state.datasets[secondary_dataset]
        
        suggested_keys = suggest_keys(primary_df, secondary_df)
        
        st.markdown("#### Suggested Join Keys")
        st.caption("Click on a suggestion to join datasets using these keys")
        
        # Display suggested keys as clickable chips
        if suggested_keys:
            cols = st.columns(3)
            for i, (primary_key, secondary_key, score) in enumerate(suggested_keys):
                with cols[i % 3]:
                    join_option = f"{primary_key} ↔️ {secondary_key} (Score: {score:.2f})"
                    if st.button(join_option, key=f"join_{i}", use_container_width=True):
                        # Perform the join operation
                        try:
                            with st.spinner(f"Joining datasets on {primary_key} and {secondary_key}..."):
                                merged_df = primary_df.merge(
                                    secondary_df, 
                                    left_on=primary_key, 
                                    right_on=secondary_key,
                                    how='inner',
                                    suffixes=('', f'_{secondary_dataset}')
                                )
                                
                                # Add the merged dataset to session state
                                merged_name = f"{primary_dataset}_{secondary_dataset}_merged"
                                st.session_state.datasets[merged_name] = merged_df
                                st.session_state.active_dataset = merged_name
                                
                                # Show success message
                                st.success(f"Successfully merged datasets into '{merged_name}'")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error joining datasets: {str(e)}")
        else:
            st.info("No common keys found that meet the matching threshold. Try custom join instead.")

    # Custom join option
    with st.expander("Custom Join"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            primary_key = st.selectbox(
                "Primary Dataset Key:",
                options=primary_df.columns if 'primary_df' in locals() else []
            )
        
        with col2:
            join_type = st.selectbox(
                "Join Type:",
                options=["Inner Join", "Left Join", "Right Join", "Outer Join"],
                index=0
            )
        
        with col3:
            secondary_key = st.selectbox(
                "Secondary Dataset Key:",
                options=secondary_df.columns if 'secondary_df' in locals() else []
            )
        
        join_mapping = {
            "Inner Join": "inner",
            "Left Join": "left",
            "Right Join": "right",
            "Outer Join": "outer"
        }
        
        if st.button("Execute Custom Join", use_container_width=True):
            try:
                with st.spinner("Performing custom join..."):
                    merged_df = primary_df.merge(
                        secondary_df,
                        left_on=primary_key,
                        right_on=secondary_key,
                        how=join_mapping[join_type],
                        suffixes=('', f'_{secondary_dataset}')
                    )
                    
                    # Add the merged dataset to session state
                    merged_name = f"{primary_dataset}_{secondary_dataset}_custom"
                    st.session_state.datasets[merged_name] = merged_df
                    st.session_state.active_dataset = merged_name
                    
                    # Show success message
                    st.success(f"Successfully created '{merged_name}' with {merged_df.shape[0]} rows")
                    st.rerun()
            except Exception as e:
                st.error(f"Error in custom join: {str(e)}")

    # Active dataset preview
    if st.session_state.active_dataset:
        st.markdown("---")
        st.subheader(f"🔍 Active Dataset: {st.session_state.active_dataset}")
        
        active_df = st.session_state.datasets[st.session_state.active_dataset]
        
        # Dataset statistics tabs
        tab1, tab2, tab3 = st.tabs(["Preview", "Summary Stats", "Column Info"])
        
        with tab1:
            st.dataframe(active_df.head(10), use_container_width=True)
            
        with tab2:
            st.dataframe(active_df.describe(include='all').T, use_container_width=True)
            
        with tab3:
            col_info = pd.DataFrame({
                'Data Type': active_df.dtypes,
                'Unique Values': active_df.nunique(),
                'Missing Values': active_df.isna().sum(),
                'Missing (%)': (active_df.isna().sum() / len(active_df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)