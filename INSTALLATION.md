# Installation Guide for Streamlit Data Analytics Platform

This installation guide will help you set up and run the Streamlit Data Analytics Platform locally on your machine.

## Prerequisites

- Python 3.9+ installed on your system
- pip (Python package installer)
- Virtual environment tool (optional but recommended)

## Step 1: Clone or Download the Project

Download and extract the project zip file to your local machine, or clone the repository if it's available on GitHub.

## Step 2: Create a Virtual Environment (Recommended)

Create a virtual environment to keep dependencies isolated:

```bash
# Using venv (Python's built-in virtual environment)
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Required Dependencies

Install all the required dependencies using pip:

```bash
pip install -r local_requirements.txt
```

Here's a list of the required packages (if local_requirements.txt is not available):

```
streamlit>=1.31.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.18.0
matplotlib>=3.8.0
reportlab>=4.0.0
camelot-py>=0.11.0
ghostscript>=0.7
pytesseract>=0.3.10
opencv-python>=4.8.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
seaborn>=0.13.0
statsmodels>=0.14.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.21.0
langchain>=0.0.350
pycairo>=1.25.0
```

## Step 4: Run the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

The application should now be running locally and accessible at [http://localhost:8501](http://localhost:8501) in your web browser.

## Optional Configurations

### Custom Port

To run the application on a specific port (e.g., 5000):

```bash
streamlit run app.py --server.port 5000
```

### Setting up OCR Capabilities

For OCR features to work:

1. Install Tesseract OCR on your system:
   - Windows: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

2. Make sure the Tesseract executable is in your PATH or set the path in your code.

### PDF Extraction Features

For PDF extraction features to work correctly:

1. Ensure Ghostscript is installed on your system:
   - Windows: Download and install from [https://www.ghostscript.com/download/gsdnld.html](https://www.ghostscript.com/download/gsdnld.html)
   - macOS: `brew install ghostscript`
   - Linux: `sudo apt-get install ghostscript`

2. Install additional dependencies for camelot:
   - Windows: Install Tk and download Ghostscript
   - macOS/Linux: `pip install tkinter`

## Troubleshooting

### Font Issues in PDF Generation
If you encounter font-related issues with PDF generation, verify that standard fonts are available on your system.

### OCR or PDF Extraction Issues
For issues with OCR or PDF extraction, make sure Tesseract and Ghostscript are correctly installed and can be found in your system path.

### Import Errors
If you encounter import errors, ensure all required packages are correctly installed:
```bash
pip install --upgrade -r local_requirements.txt
```

### Permission Issues
If you encounter permission issues when accessing files:
- On Linux/macOS: `chmod +x app.py`
- On Windows: Make sure you run as administrator if needed

## Support

If you encounter any issues, please refer to the documentation or contact support.