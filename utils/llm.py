import os
import streamlit as st
import json
import pandas as pd
import requests

# Try to import langchain, but gracefully handle if it's not available
try:
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

@st.cache_resource
def get_llm_instance(_key=None):
    """
    Get an LLM instance based on the configuration in session state.
    Uses caching to avoid reinitializing the model unnecessarily.
    
    Parameters:
    -----------
    _key : str, optional
        Optional API key override
    
    Returns:
    --------
    LLM
        Configured LLM instance (OpenAI API or local model)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required but not installed.")
        
    # If config doesn't exist in session state, use defaults
    if 'llm_config' not in st.session_state:
        st.session_state.llm_config = {
            "type": "OpenAI API",
            "key": os.getenv("OPENAI_API_KEY", ""),
            "model_path": ""
        }
    
    config = st.session_state.llm_config
    
    # Use API key from parameters if provided, otherwise from config
    api_key = _key if _key else config.get("key", os.getenv("OPENAI_API_KEY", ""))
    
    if config["type"] == "OpenAI API":
        # Validate API key exists
        if not api_key:
            raise ValueError("OpenAI API key is not provided. Please add it in the sidebar settings.")
            
        # Use OpenAI API
        return OpenAI(
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-instruct",
            temperature=0.1
        )
    else:
        # Use local model
        model_path = config.get("model_path", "")
        if not model_path:
            raise ValueError("Local model path is not specified")
        
        try:
            # Import HuggingFaceHub or other local model libraries as needed
            from langchain.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            # Load model and tokenizer from local path
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Create pipeline
            llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512
            )
            
            # Create and return LLM instance
            return HuggingFacePipeline(pipeline=llm_pipeline)
        except ImportError:
            raise ImportError("Required modules for local models are not installed.")

def analysis_chain(data_summary):
    """
    Generate insights about a dataset using an LLM.
    
    Parameters:
    -----------
    data_summary : dict
        Summary statistics about the dataset
    
    Returns:
    --------
    list
        List of insight strings
    """
    if not LANGCHAIN_AVAILABLE:
        return ["LangChain is required but not installed."]
        
    try:
        # Get LLM instance
        llm = get_llm_instance()
        
        # Define prompt template for analysis
        analysis_template = """
        You are a data analyst tasked with providing insights about a dataset.
        
        DATASET INFORMATION:
        - Columns: {columns}
        - Summary Statistics: {stats}
        - Data Types: {dtypes}
        - Sample Size: {sample_size}
        
        Based on the information above, provide 3 concise, valuable business insights about this dataset.
        Focus on patterns, outliers, or interesting relationships that might be present.
        
        Format your response as a list of 3 bullet points (1-2 sentences each).
        """
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["columns", "stats", "dtypes", "sample_size"],
            template=analysis_template
        )
        
        # Convert stats dictionary to a readable string
        stats_str = json.dumps(data_summary["stats"])
        dtypes_str = json.dumps(data_summary["dtypes"])
        
        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(
            columns=data_summary["columns"],
            stats=stats_str,
            dtypes=dtypes_str,
            sample_size=data_summary["sample_size"]
        )
        
        # Parse the bullet points
        insights = [insight.strip().replace('- ', '') for insight in response.split('\n') if insight.strip()]
        
        # Limit to 3 insights
        return insights[:3]
    
    except Exception as e:
        # Return error as an insight
        return [f"Error generating insights: {str(e)}"]

def generate_answer(question, schema, sample_rows):
    """
    Generate an answer to a natural language question about the dataset.
    Also returns chart code if applicable.
    
    Parameters:
    -----------
    question : str
        The user's question about the data
    schema : dict
        Information about the dataset schema
    sample_rows : list
        Sample rows from the dataset
    
    Returns:
    --------
    dict
        Dictionary with answer text and optional chart code
    """
    if not LANGCHAIN_AVAILABLE:
        return {"answer": "LangChain is required but not installed.", "chart_code": None}
        
    try:
        # Get LLM instance
        llm = get_llm_instance()
        
        # Define prompt template for query analysis
        query_template = """
        You are an expert data analyst tasked with answering questions about a dataset.
        
        DATASET SCHEMA:
        - Columns: {columns}
        - Data Types: {dtypes}
        - Unique Value Counts: {unique_counts}
        - Sample Size: {sample_size}
        
        SAMPLE ROWS (for reference only):
        {sample_rows}
        
        USER QUESTION: {question}
        
        First, provide a clear and concise answer to the user's question, incorporating relevant data insights.
        
        Then, provide Python code to create a Plotly Express visualization that best answers the question. Use ONLY Plotly Express (px).
        The code should assume the data is in a pandas DataFrame called 'df'. Keep the code simple and ensure it produces a meaningful visualization.
        
        Format your response as JSON with the following structure:
        {{"answer": "Your answer here", "chart_code": "import plotly.express as px\\n...your code here...", "chart_type": "Bar/Line/Scatter/etc"}}
        """
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["columns", "dtypes", "unique_counts", "sample_size", "sample_rows", "question"],
            template=query_template
        )
        
        # Prepare sample rows string (but limit sensitive data)
        sample_rows_str = json.dumps(sample_rows, indent=2)
        
        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(
            columns=schema["columns"],
            dtypes=json.dumps(schema["dtypes"]),
            unique_counts=json.dumps(schema["unique_counts"]),
            sample_size=schema["sample_size"],
            sample_rows=sample_rows_str,
            question=question
        )
        
        # Parse the JSON response
        # Extract the JSON part from the response (in case there's extra text)
        response = response.strip()
        
        # Find the beginning and end of the JSON object
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        else:
            # If JSON parsing fails, return just the text as an answer
            return {"answer": response, "chart_code": None}
    
    except Exception as e:
        # Return error message
        return {"answer": f"Error generating answer: {str(e)}", "chart_code": None}