"""
Data Loader Module

This module provides high-level data loading functionality for the hypothesis testing project.
It handles loading all required datasets with proper error handling and validation.
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path

from .file_reader import FileReader

# Try to import test config first, fall back to regular config
try:
    import test_config as config
    print("Using test configuration")
except ImportError:
    import config
    print("Using production configuration")


class DataLoader:
    """
    Main data loader class that handles loading all datasets required for analysis.
    """
    
    def __init__(self):
        """Initialize the DataLoader with a FileReader instance."""
        self.file_reader = FileReader(suppress_warnings=True)
        self._loaded_data = {}
    
    def load_main_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load the main datasets required for analysis.
        
        Returns:
            Tuple containing (lp_history_df, salary_df, performance_df, punishment_df, president_cup_df)
        """
        print("Loading main datasets...")
        
        # Load LP history data
        lp_history_df = self.file_reader.read_csv(config.get_data_path("lp_history"))

        # Load salary data
        salary_df = self.file_reader.read_csv(config.get_data_path("salary"))

        # Load performance data
        performance_df = self.file_reader.read_csv(config.get_data_path("performance"))

        # Load punishment data
        punishment_df = self.file_reader.read_csv(config.get_data_path("punishment"))

        # Load president cup data
        president_cup_df = self.file_reader.read_csv(config.get_data_path("president_cup"))
        
        # Store loaded data for potential reuse
        self._loaded_data.update({
            "lp_history": lp_history_df,
            "salary": salary_df,
            "performance": performance_df,
            "punishment": punishment_df,
            "president_cup": president_cup_df
        })
        
        print("Main datasets loaded successfully!")
        return lp_history_df, salary_df, performance_df, punishment_df, president_cup_df
    
    def load_jimu_miss_data(self) -> pd.DataFrame:
        """
        Load the jimu miss (administrative error) data.
        
        Returns:
            DataFrame containing jimu miss data
        """
        print("Loading jimu miss data...")
        jimu_miss_df = self.file_reader.read_excel(config.get_data_path("jimu_miss"))
        self._loaded_data["jimu_miss"] = jimu_miss_df
        return jimu_miss_df
    
    def load_answer_record_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load answer record data from multiple sheets.
        
        Returns:
            Tuple containing three DataFrames from different sheets
        """
        print("Loading answer record data...")
        file_path = config.get_data_path("answer_record")
        
        answer_record_df_1 = self.file_reader.read_excel(file_path, sheet_name=0)
        answer_record_df_2 = self.file_reader.read_excel(file_path, sheet_name=1)
        answer_record_df_3 = self.file_reader.read_excel(file_path, sheet_name=2)
        
        self._loaded_data.update({
            "answer_record_1": answer_record_df_1,
            "answer_record_2": answer_record_df_2,
            "answer_record_3": answer_record_df_3
        })
        
        return answer_record_df_1, answer_record_df_2, answer_record_df_3
    
    def load_complaint_data(self) -> pd.DataFrame:
        """
        Load complaint data.
        
        Returns:
            DataFrame containing complaint data
        """
        print("Loading complaint data...")
        complaint_df = self.file_reader.read_excel(config.get_data_path("complaint"), sheet_name=0)
        self._loaded_data["complaint"] = complaint_df
        return complaint_df
    
    def load_u24_247_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load U24-247 data from two files.
        
        Returns:
            Tuple containing two DataFrames
        """
        print("Loading U24-247 data...")
        u24_247_df_1 = self.file_reader.read_excel(config.get_data_path("u24_247_1"))
        u24_247_df_2 = self.file_reader.read_excel(config.get_data_path("u24_247_2"))
        
        self._loaded_data.update({
            "u24_247_1": u24_247_df_1,
            "u24_247_2": u24_247_df_2
        })
        
        return u24_247_df_1, u24_247_df_2
    
    def load_meeting_attendance_data(self) -> pd.DataFrame:
        """
        Load meeting attendance data.
        
        Returns:
            DataFrame containing meeting attendance data
        """
        print("Loading meeting attendance data...")
        meeting_attendance_df = self.file_reader.read_csv(config.get_data_path("meeting_attendance"))
        self._loaded_data["meeting_attendance"] = meeting_attendance_df
        return meeting_attendance_df
    
    def get_loaded_data(self, key: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get previously loaded data.
        
        Args:
            key: Specific dataset key to retrieve, or None for all data
            
        Returns:
            Dictionary containing requested datasets
        """
        if key is None:
            return self._loaded_data.copy()
        
        if key not in self._loaded_data:
            raise KeyError(f"Dataset '{key}' has not been loaded yet")
        
        return {key: self._loaded_data[key]}
    
    def get_data_summary(self) -> Dict[str, Dict[str, int]]:
        """
        Get summary information about loaded datasets.
        
        Returns:
            Dictionary containing dataset summaries
        """
        summary = {}
        for key, df in self._loaded_data.items():
            summary[key] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            }
        
        return summary
