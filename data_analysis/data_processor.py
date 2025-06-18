"""
Data Processor Module

This module handles data processing operations including cleaning,
transformation, and preparation for analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime

from config import PROCESSING_CONSTANTS, OUTPUT_CONFIG


class DataProcessor:
    """
    A class for handling data processing and transformation operations.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        self.year_offset = PROCESSING_CONSTANTS["year_offset"]
        self.filter_year_start = PROCESSING_CONSTANTS["filter_year_start"]
        self.date_format = OUTPUT_CONFIG["date_format"]
        self.datetime_format = OUTPUT_CONFIG["datetime_format"]
    
    def process_date_columns(self, df: pd.DataFrame, date_config: Dict[str, str]) -> pd.DataFrame:
        """
        Process date columns in a DataFrame.
        
        Args:
            df: DataFrame to process
            date_config: Configuration for date column processing
            
        Returns:
            DataFrame with processed date columns
        """
        df_processed = df.copy()
        
        # Apply year offset if specified
        if "year" in date_config and date_config["year"] in df_processed.columns:
            year_col = date_config["year"]
            df_processed[year_col] = df_processed[year_col] + self.year_offset
            print(f"Applied year offset (+{self.year_offset}) to column: {year_col}")
        
        return df_processed
    
    def create_date_column(self, df: pd.DataFrame, year_col: str, month_col: str, 
                          day_col: str, new_col_name: str = "Date") -> pd.DataFrame:
        """
        Create a date column from separate year, month, and day columns.
        
        Args:
            df: DataFrame to process
            year_col: Year column name
            month_col: Month column name
            day_col: Day column name
            new_col_name: Name for the new date column
            
        Returns:
            DataFrame with new date column
        """
        df_processed = df.copy()
        
        try:
            df_processed[new_col_name] = pd.to_datetime(
                dict(
                    year=df_processed[year_col],
                    month=df_processed[month_col],
                    day=df_processed[day_col]
                )
            )
            print(f"Created date column '{new_col_name}' from {year_col}, {month_col}, {day_col}")
        except Exception as e:
            print(f"Error creating date column: {str(e)}")
            raise
        
        return df_processed
    
    def process_datetime_column(self, df: pd.DataFrame, column: str, 
                              format_str: Optional[str] = None) -> pd.DataFrame:
        """
        Process a datetime column with proper formatting.
        
        Args:
            df: DataFrame to process
            column: Column name to process
            format_str: DateTime format string (if None, uses default)
            
        Returns:
            DataFrame with processed datetime column
        """
        df_processed = df.copy()
        
        if format_str is None:
            format_str = self.datetime_format
        
        try:
            df_processed[column] = pd.to_datetime(df_processed[column], format=format_str)
            print(f"Processed datetime column: {column}")
        except Exception as e:
            # Try automatic parsing if format fails
            try:
                df_processed[column] = pd.to_datetime(df_processed[column])
                print(f"Processed datetime column with automatic parsing: {column}")
            except Exception as e2:
                print(f"Error processing datetime column {column}: {str(e2)}")
                raise
        
        return df_processed
    
    def extract_year_month(self, df: pd.DataFrame, date_column: str, 
                          year_col_name: str = "year", month_col_name: str = "month") -> pd.DataFrame:
        """
        Extract year and month from a datetime column.
        
        Args:
            df: DataFrame to process
            date_column: Source datetime column
            year_col_name: Name for the year column
            month_col_name: Name for the month column
            
        Returns:
            DataFrame with extracted year and month columns
        """
        df_processed = df.copy()
        
        # Ensure the column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_processed[date_column]):
            df_processed = self.process_datetime_column(df_processed, date_column)
        
        df_processed[year_col_name] = df_processed[date_column].dt.year
        df_processed[month_col_name] = df_processed[date_column].dt.month
        
        print(f"Extracted year and month from {date_column}")
        return df_processed
    
    def clean_dataframe(self, df: pd.DataFrame, drop_columns: Optional[List[str]] = None,
                       drop_na: bool = True, fill_na_value: Any = None) -> pd.DataFrame:
        """
        Clean a DataFrame by dropping columns and handling missing values.
        
        Args:
            df: DataFrame to clean
            drop_columns: List of columns to drop
            drop_na: Whether to drop rows with missing values
            fill_na_value: Value to fill missing values with (if not dropping)
            
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        
        # Drop specified columns
        if drop_columns:
            existing_columns = [col for col in drop_columns if col in df_cleaned.columns]
            if existing_columns:
                df_cleaned = df_cleaned.drop(existing_columns, axis=1)
                print(f"Dropped columns: {existing_columns}")
        
        # Handle missing values
        if drop_na:
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            final_rows = len(df_cleaned)
            print(f"Dropped {initial_rows - final_rows} rows with missing values")
        elif fill_na_value is not None:
            df_cleaned = df_cleaned.fillna(fill_na_value)
            print(f"Filled missing values with: {fill_na_value}")
        
        return df_cleaned
    
    def filter_by_year(self, df: pd.DataFrame, year_column: str, 
                      start_year: Optional[int] = None) -> pd.DataFrame:
        """
        Filter DataFrame by year.
        
        Args:
            df: DataFrame to filter
            year_column: Column containing year values
            start_year: Minimum year to include (if None, uses default)
            
        Returns:
            Filtered DataFrame
        """
        if start_year is None:
            start_year = self.filter_year_start
        
        df_filtered = df[df[year_column] >= start_year].copy()
        
        initial_rows = len(df)
        final_rows = len(df_filtered)
        print(f"Filtered by year >= {start_year}: {initial_rows} -> {final_rows} rows")
        
        return df_filtered
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: DataFrame to process
            subset: Columns to consider for identifying duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        df_deduped = df.drop_duplicates(subset=subset)
        final_rows = len(df_deduped)
        
        print(f"Removed duplicates: {initial_rows} -> {final_rows} rows")
        return df_deduped
    
    def get_fiscal_half(self, row: pd.Series) -> str:
        """
        Calculate fiscal half for a given row.
        
        Args:
            row: DataFrame row containing S_YR and S_MO columns
            
        Returns:
            Fiscal half string (e.g., "2023_H1")
        """
        if 4 <= row['S_MO'] <= 9:
            return f"{row['S_YR']}_H1"  # Fiscal first half
        else:
            # Fiscal second half
            return f"{row['S_YR'] - 1}_H2" if row['S_MO'] < 4 else f"{row['S_YR']}_H2"
    
    def add_fiscal_half_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fiscal half column to DataFrame.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with FISCAL_HALF column added
        """
        df_processed = df.copy()
        df_processed['FISCAL_HALF'] = df_processed.apply(self.get_fiscal_half, axis=1)
        print("Added FISCAL_HALF column")
        return df_processed
