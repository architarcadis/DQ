import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
from utils.llm import generate_answer

def show_ask_the_data():
    # Main content
    st.subheader("💬 Ask the Data")
    st.markdown("Query your data using natural language")

    # Check if datasets exist
    if not st.session_state.datasets:
        st.info("👆 Start by uploading some data files using the sidebar", icon="ℹ️")
        
        # Using a stock photo for visual appeal
        st.image("https://pixabay.com/get/g00b95e1d5ef09c8e219e73f6a50452bb18e6a8fc74c1a1cb4ec33bf6b0c6ccd2aef88d99d883f9a9626ce44f46d0df88c15adbb33bbe21f16cfb1d2bcf4de9c1_1280.jpg", 
                caption="Ask questions about your data in natural language", 
                width=500)
        return

    # Initialize conversation history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # API Key verification
    if st.session_state.llm_config['type'] == 'OpenAI API' and not st.session_state.llm_config['key']:
        st.warning("⚠️ Please add your OpenAI API key in the sidebar settings to use this feature.")
        return

    # Chat interface
    st.subheader("Ask a question about your data")

    # Select dataset
    dataset_name = st.selectbox(
        "Select a dataset to query:",
        options=list(st.session_state.datasets.keys()),
        index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) 
            if st.session_state.active_dataset in st.session_state.datasets 
            else 0
    )

    # Get the dataset
    df = st.session_state.datasets[dataset_name]

    # Display dataset preview
    with st.expander("Dataset Preview"):
        st.dataframe(df.head(5), use_container_width=True)

    # Example questions
    st.markdown("#### Example questions:")
    examples_col1, examples_col2 = st.columns(2)

    example_questions = [
        "What are the top 5 values in column X?",
        "Show me the relationship between X and Y",
        "What is the average of column Z?",
        "Create a bar chart of X by category Y",
        "What percentage of records have Y > 100?",
        "Which category has the highest average X?"
    ]

    # Replace generic column names with actual column names
    if not df.empty:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if numeric_cols and categorical_cols:
            for i, q in enumerate(example_questions):
                q = q.replace('X', numeric_cols[0] if numeric_cols else 'value')
                q = q.replace('Y', categorical_cols[0] if categorical_cols else 'category')
                q = q.replace('Z', numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] if numeric_cols else 'value')
                example_questions[i] = q

    for i, q in enumerate(example_questions):
        if i < 3:
            if examples_col1.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.user_question = q
        else:
            if examples_col2.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.user_question = q

    # Input for user question
    user_question = st.text_input(
        "Your question:",
        value=st.session_state.get('user_question', ''),
        placeholder="e.g., What is the average sales amount?",
        key="question_input"
    )

    # Submit button
    submit = st.button("Ask Question", use_container_width=True)

    # Process the question when submitted
    if submit and user_question:
        # Save question to session state
        st.session_state.user_question = user_question
        
        # Create schema information for the LLM
        schema = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "unique_counts": df.nunique().to_dict(),
            "sample_size": len(df)
        }
        
        # Sample rows as context (limit to 5 rows)
        sample_rows = df.head(5).to_dict(orient='records')
        
        with st.spinner("Analyzing your question..."):
            try:
                # Call LLM to generate answer
                result = generate_answer(user_question, schema, sample_rows)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": result.get("answer", "I couldn't generate an answer."),
                    "chart_code": result.get("chart_code", None),
                    "chart_type": result.get("chart_type", None)
                })
                
                # Clear input after submission
                st.session_state.user_question = ""
                
                # Rerun to update UI
                st.rerun()
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            st.markdown("---")
            
            # Question
            st.markdown(f"**🙋‍♂️ Question:** {item['question']}")
            
            # Answer
            st.markdown(f"**🤖 Answer:** {item['answer']}")
            
            # Render chart if chart code is available
            if item['chart_code']:
                try:
                    # Extract the code block from the response
                    code = item['chart_code']
                    
                    # Create a namespace
                    namespace = {"df": df, "px": px, "pd": pd}
                    
                    # Execute the code to get the figure
                    exec_globals = {}
                    exec("import plotly.express as px\nimport pandas as pd", exec_globals)
                    exec_globals["df"] = df
                    
                    # Execute the code
                    exec(f"fig = {code}", exec_globals)
                    
                    # Get the figure from the namespace
                    fig = exec_globals.get("fig")
                    
                    if fig:
                        # Update layout for better appearance
                        fig.update_layout(
                            margin=dict(l=20, r=20, t=40, b=20),
                            title_font=dict(size=16, family="Montserrat", color="#333"),
                            font=dict(family="Montserrat"),
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Could not generate chart from the code.")
                except Exception as e:
                    st.error(f"Error rendering chart: {str(e)}")
                    with st.expander("Chart code (for debugging)"):
                        st.code(code)
                    
            # Delete button
            if st.button("🗑️ Delete", key=f"delete_{i}"):
                st.session_state.chat_history.pop(-(i+1))
                st.rerun()
    else:
        st.info("Ask a question to get started!")

    # Clear history button
    if st.session_state.chat_history:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()