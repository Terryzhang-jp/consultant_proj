"""
Data Merger Module

This module handles merging operations between different datasets
for the hypothesis testing project.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Any


class DataMerger:
    """
    A class for handling data merging operations across multiple datasets.
    """
    
    def __init__(self):
        """Initialize the DataMerger."""
        self.merge_history = []
    
    def merge_dataframes(self, left_df: pd.DataFrame, right_df: pd.DataFrame,
                        left_on: List[str], right_on: List[str],
                        how: str = 'left', suffixes: Tuple[str, str] = ('_x', '_y')) -> pd.DataFrame:
        """
        Merge two DataFrames with detailed logging.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            left_on: Columns to join on from left DataFrame
            right_on: Columns to join on from right DataFrame
            how: Type of merge to perform
            suffixes: Suffixes for overlapping column names
            
        Returns:
            Merged DataFrame
        """
        # Record initial sizes
        left_size = len(left_df)
        right_size = len(right_df)
        
        # Perform merge
        merged_df = pd.merge(
            left_df, right_df,
            left_on=left_on, right_on=right_on,
            how=how, suffixes=suffixes
        )
        
        # Record merge information
        merge_info = {
            "left_size": left_size,
            "right_size": right_size,
            "merged_size": len(merged_df),
            "left_on": left_on,
            "right_on": right_on,
            "how": how,
            "suffixes": suffixes
        }
        self.merge_history.append(merge_info)
        
        print(f"Merge completed: {left_size} + {right_size} -> {len(merged_df)} rows ({how} join)")
        
        return merged_df
    
    def merge_lp_history_with_salary(self, lp_history_df: pd.DataFrame, 
                                   salary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge LP history data with salary data.
        
        Args:
            lp_history_df: LP history DataFrame
            salary_df: Salary DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging LP history with salary data...")
        
        # Perform merge
        merged_df = self.merge_dataframes(
            lp_history_df, salary_df,
            left_on=['LP', 'RANK', 'S_YR', 'S_MO'],
            right_on=['LP', 'RANK', 'S_YR', 'S_MO'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_performance_data(self, base_df: pd.DataFrame, 
                                  performance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with performance data.
        
        Args:
            base_df: Base DataFrame
            performance_df: Performance DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with performance data...")
        
        merged_df = self.merge_dataframes(
            base_df, performance_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'year', 'month'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_punishment_data(self, base_df: pd.DataFrame, 
                                 punishment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with punishment data.
        
        Args:
            base_df: Base DataFrame
            punishment_df: Punishment DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with punishment data...")
        
        merged_df = self.merge_dataframes(
            base_df, punishment_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'year', 'month'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_president_cup_data(self, base_df: pd.DataFrame, 
                                    president_cup_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with president cup data.
        
        Args:
            base_df: Base DataFrame
            president_cup_df: President cup DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with president cup data...")
        
        merged_df = self.merge_dataframes(
            base_df, president_cup_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'CONTEST_YYY', 'CONTEST_MM'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_jimu_miss_data(self, base_df: pd.DataFrame, 
                                jimu_miss_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with jimu miss data.
        
        Args:
            base_df: Base DataFrame
            jimu_miss_df: Jimu miss DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with jimu miss data...")
        
        merged_df = self.merge_dataframes(
            base_df, jimu_miss_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'year', 'month'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_complaint_data(self, base_df: pd.DataFrame, 
                                complaint_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with complaint data.
        
        Args:
            base_df: Base DataFrame
            complaint_df: Complaint DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with complaint data...")
        
        merged_df = self.merge_dataframes(
            base_df, complaint_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'year', 'month'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_u24_247_data(self, base_df: pd.DataFrame, 
                               u24_247_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with U24-247 data.
        
        Args:
            base_df: Base DataFrame
            u24_247_df: U24-247 DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with U24-247 data...")
        
        merged_df = self.merge_dataframes(
            base_df, u24_247_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'year', 'month'],
            how='left'
        )
        
        return merged_df
    
    def merge_with_meeting_attendance_data(self, base_df: pd.DataFrame, 
                                         meeting_attendance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge base DataFrame with meeting attendance data.
        
        Args:
            base_df: Base DataFrame
            meeting_attendance_df: Meeting attendance DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging with meeting attendance data...")
        
        merged_df = self.merge_dataframes(
            base_df, meeting_attendance_df,
            left_on=['LP', 'S_YR', 'S_MO'],
            right_on=['LP', 'year', 'month'],
            how='left'
        )
        
        return merged_df
    
    def get_merge_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of merge operations.
        
        Returns:
            List of merge operation details
        """
        return self.merge_history.copy()
    
    def get_merge_summary(self) -> Dict[str, Any]:
        """
        Get summary of all merge operations.
        
        Returns:
            Dictionary containing merge summary
        """
        if not self.merge_history:
            return {"total_merges": 0}
        
        total_merges = len(self.merge_history)
        merge_types = [merge["how"] for merge in self.merge_history]
        
        return {
            "total_merges": total_merges,
            "merge_types": pd.Series(merge_types).value_counts().to_dict(),
            "average_result_size": sum(merge["merged_size"] for merge in self.merge_history) / total_merges
        }
