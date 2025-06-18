"""
Data Mapper Module

This module handles data mapping operations, including creating mappings
for categorical variables and applying them across datasets.
"""

import pandas as pd
from typing import Dict, Tuple, Any, List


class DataMapper:
    """
    A class for handling data mapping operations across datasets.
    """
    
    def __init__(self):
        """Initialize the DataMapper."""
        self.mappings = {}
        self.office_mappings = {}
    
    def map_column_with_prefix(self, df: pd.DataFrame, column: str, 
                              prefix: str, mapping_dict: Dict[Any, str]) -> Tuple[pd.DataFrame, Dict[Any, str]]:
        """
        Create a mapping for a column with a specified prefix.
        
        Args:
            df: DataFrame to process
            column: Column name to map
            prefix: Prefix for the mapped values
            mapping_dict: Existing mapping dictionary to update
            
        Returns:
            Tuple of (updated DataFrame, updated mapping dictionary)
        """
        mapping_list = []
        
        # Get unique values from the column
        unique_values = df[column].values
        counter = len(mapping_dict)
        
        for value in unique_values:
            if value not in mapping_dict:
                # Create new mapping with prefix
                mapped_value = f"{prefix}_{counter}"
                mapping_dict[value] = mapped_value
                mapping_list.append(mapped_value)
                counter += 1
            else:
                # Use existing mapping
                mapping_list.append(mapping_dict[value])
        
        # Apply mapping to the DataFrame
        df_copy = df.copy()
        df_copy[column] = mapping_list
        
        return df_copy, mapping_dict
    
    def apply_lp_mappings(self, lp_history_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Any, str]]:
        """
        Apply LP (Life Planner) mappings to the LP history DataFrame.
        
        Args:
            lp_history_df: LP history DataFrame
            
        Returns:
            Tuple of (processed DataFrame, LP mappings dictionary)
        """
        print("Applying LP mappings...")
        
        df_processed = lp_history_df.copy()
        lp_mappings = {}
        
        # Map LP column
        df_processed, lp_mappings = self.map_column_with_prefix(
            df_processed, 'LP', 'LP', lp_mappings
        )
        
        # Map MGR (Manager) column
        df_processed, lp_mappings = self.map_column_with_prefix(
            df_processed, 'MGR', 'LP', lp_mappings
        )
        
        # Map AMGR (Area Manager) column
        df_processed, lp_mappings = self.map_column_with_prefix(
            df_processed, 'AMGR', 'LP', lp_mappings
        )
        
        self.mappings = lp_mappings
        print(f"LP mappings created: {len(lp_mappings)} unique mappings")
        
        return df_processed, lp_mappings
    
    def apply_office_mappings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Any, str]]:
        """
        Apply office mappings to the DataFrame.
        
        Args:
            df: DataFrame containing office information
            
        Returns:
            Tuple of (processed DataFrame, office mappings dictionary)
        """
        print("Applying office mappings...")
        
        df_processed, office_mappings = self.map_column_with_prefix(
            df, 'OFFICE', 'OFFICE', {}
        )
        
        self.office_mappings = office_mappings
        print(f"Office mappings created: {len(office_mappings)} unique mappings")
        
        return df_processed, office_mappings
    
    def apply_mappings_to_dataframe(self, df: pd.DataFrame, column: str, 
                                  mapping_dict: Dict[Any, str], new_column: str = None) -> pd.DataFrame:
        """
        Apply existing mappings to a DataFrame column.
        
        Args:
            df: DataFrame to process
            column: Source column name
            mapping_dict: Mapping dictionary to apply
            new_column: New column name (if None, uses 'LP')
            
        Returns:
            DataFrame with applied mappings
        """
        if new_column is None:
            new_column = 'LP'
        
        df_copy = df.copy()
        df_copy[new_column] = df_copy[column].map(mapping_dict)
        
        # Report mapping success rate
        mapped_count = df_copy[new_column].notna().sum()
        total_count = len(df_copy)
        success_rate = mapped_count / total_count if total_count > 0 else 0
        
        print(f"Mapping applied to {column}: {mapped_count}/{total_count} ({success_rate:.2%} success rate)")
        
        return df_copy
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current mappings.
        
        Returns:
            Dictionary containing mapping statistics
        """
        return {
            "lp_mappings_count": len(self.mappings),
            "office_mappings_count": len(self.office_mappings),
            "total_mappings": len(self.mappings) + len(self.office_mappings)
        }
    
    def export_mappings(self) -> Dict[str, Dict[Any, str]]:
        """
        Export all mappings for external use.
        
        Returns:
            Dictionary containing all mapping dictionaries
        """
        return {
            "lp_mappings": self.mappings.copy(),
            "office_mappings": self.office_mappings.copy()
        }
