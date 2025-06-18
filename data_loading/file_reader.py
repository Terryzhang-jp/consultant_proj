"""
File Reader Module

This module provides utilities for reading various file formats
used in the hypothesis testing project.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings


class FileReader:
    """
    A utility class for reading different file formats with consistent error handling.
    """
    
    def __init__(self, suppress_warnings: bool = True):
        """
        Initialize the FileReader.
        
        Args:
            suppress_warnings: Whether to suppress pandas warnings
        """
        self.suppress_warnings = suppress_warnings
        if suppress_warnings:
            warnings.filterwarnings("ignore")
    
    def read_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Read a CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv
            
        Returns:
            DataFrame containing the CSV data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, **kwargs)
            print(f"Successfully loaded CSV: {file_path.name} ({len(df)} rows)")
            return df
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError(f"CSV file is empty: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading CSV file {file_path}: {str(e)}")
    
    def read_excel(self, file_path: Union[str, Path], sheet_name: Union[str, int, List] = 0, **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Read an Excel file with error handling.
        
        Args:
            file_path: Path to the Excel file
            sheet_name: Sheet name(s) to read
            **kwargs: Additional arguments to pass to pd.read_excel
            
        Returns:
            DataFrame or dictionary of DataFrames
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {file_path}")
        
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            
            if isinstance(df, dict):
                total_rows = sum(len(sheet_df) for sheet_df in df.values())
                print(f"Successfully loaded Excel: {file_path.name} ({len(df)} sheets, {total_rows} total rows)")
            else:
                print(f"Successfully loaded Excel: {file_path.name} ({len(df)} rows)")
            
            return df
        except Exception as e:
            raise Exception(f"Error reading Excel file {file_path}: {str(e)}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Union[str, int, float]]:
        """
        Get basic information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        return {
            "exists": True,
            "name": file_path.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix,
            "absolute_path": str(file_path.absolute())
        }
    
    def validate_file_paths(self, file_paths: List[Union[str, Path]]) -> Dict[str, bool]:
        """
        Validate multiple file paths.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary mapping file paths to existence status
        """
        validation_results = {}
        for path in file_paths:
            path_obj = Path(path)
            validation_results[str(path)] = path_obj.exists()
        
        return validation_results
