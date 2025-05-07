import pandas as pd
import io
from PIL import Image
import pytesseract
import os
import streamlit as st

# Try to import camelot, but gracefully handle if it's not available
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

@st.cache_resource
def extract_data(uploaded_files):
    """
    Extract data from uploaded files of various formats (Excel, CSV, PDF, Images).
    
    Parameters:
    -----------
    uploaded_files : list
        List of uploaded file objects from st.file_uploader
        
    Returns:
    --------
    dict
        Dictionary mapping dataset names to pandas DataFrames
    """
    datasets = {}
    
    for file in uploaded_files:
        try:
            file_ext = os.path.splitext(file.name)[1].lower()
            file_name = os.path.splitext(file.name)[0]
            
            # Process different file types
            if file_ext in ['.xlsx', '.xls']:
                # Excel files - potentially multi-sheet
                excel_file = pd.ExcelFile(file)
                sheet_names = excel_file.sheet_names
                
                # Process each sheet as a separate dataset
                for sheet in sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet)
                    
                    # Skip empty sheets
                    if not df.empty:
                        dataset_name = f"{file_name}_{sheet}" if len(sheet_names) > 1 else file_name
                        datasets[dataset_name] = df
            
            elif file_ext == '.csv':
                # CSV files
                df = pd.read_csv(file)
                datasets[file_name] = df
            
            elif file_ext == '.pdf':
                # PDF files - extract tables using camelot if available
                if CAMELOT_AVAILABLE:
                    # First save the file to a temp location since camelot needs a file path
                    with open("temp.pdf", "wb") as f:
                        f.write(file.getvalue())
                    
                    # Extract tables
                    tables = camelot.read_pdf("temp.pdf", pages="all", flavor="lattice")
                    
                    # Process each table as a separate dataset
                    for i, table in enumerate(tables):
                        df = table.df
                        
                        # Use first row as header if it looks like a header
                        if not df.empty:
                            df.columns = df.iloc[0]
                            df = df.iloc[1:].reset_index(drop=True)
                            
                            # Convert numeric columns
                            for col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(df[col])
                                except (ValueError, TypeError):
                                    pass
                            
                            dataset_name = f"{file_name}_table_{i+1}" if len(tables) > 1 else file_name
                            datasets[dataset_name] = df
                    
                    # Clean up temp file
                    if os.path.exists("temp.pdf"):
                        os.remove("temp.pdf")
                else:
                    # If camelot is not available, return a simple message
                    df = pd.DataFrame({"Message": ["PDF extraction requires camelot-py package."]})
                    datasets[f"{file_name}_error"] = df
            
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                # Image files - use OCR to extract text tables
                # This is a simple implementation and may need refinement for complex tables
                img = Image.open(file)
                
                # Extract text from image
                text = pytesseract.image_to_string(img)
                
                # Simple parsing of tabular data in the text
                # Assumes tab or multiple space separation
                lines = text.strip().split('\n')
                rows = []
                
                for line in lines:
                    # Skip empty lines
                    if line.strip():
                        # Split by multiple spaces or tabs
                        cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                        rows.append(cells)
                
                # Convert to DataFrame if we have data
                if rows:
                    # Use first row as header
                    header = rows[0]
                    data = rows[1:]
                    
                    # Ensure all rows have same number of columns
                    max_cols = max(len(row) for row in rows)
                    padded_data = []
                    
                    for row in data:
                        if len(row) < max_cols:
                            padded_data.append(row + [''] * (max_cols - len(row)))
                        else:
                            padded_data.append(row)
                    
                    # Create DataFrame
                    if padded_data:
                        df = pd.DataFrame(padded_data, columns=header)
                        datasets[file_name] = df
                    else:
                        # Fall back to just raw text if table detection fails
                        df = pd.DataFrame({'text': lines})
                        datasets[f"{file_name}_text"] = df
        
        except Exception as e:
            # Log the error but continue processing other files
            print(f"Error processing file {file.name}: {str(e)}")
            continue
    
    return datasets