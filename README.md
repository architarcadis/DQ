# Streamlit Data Analytics Platform

A comprehensive data analytics platform built with Streamlit, enabling data extraction, visualization, quality assessment, and machine learning capabilities in a user-friendly interface. The application leverages the DAMA framework for standardized data quality assessment.

## Features

- **Data Hub**: Upload, explore, combine, and transform your datasets with support for Excel, CSV, PDF, and image formats
- **Data Quality Assessment**: Comprehensive quality analysis with professional PDF reporting using DAMA framework principles
- **Robust Charts**: Library of interactive visualizations with advanced customization options
- **Advanced Analytics**: Build machine learning models with automatic feature selection and model evaluation
- **Ask the Data**: Natural language querying of your datasets with AI-generated insights

## Key Components

- **Tab-based Navigation**: Intuitive interface organized by analytics functions
- **Sidebar Components**: File upload and LLM configuration
- **Professional Reporting**: Generate comprehensive PDF reports with data quality assessments, visualizations, and recommendations
- **Interactive Visualizations**: Plotly-based interactive charts with export capabilities

## Local Installation

For detailed installation instructions, please see [INSTALLATION.md](INSTALLATION.md).

Quick start:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r local_requirements.txt

# Run the application
streamlit run app.py
