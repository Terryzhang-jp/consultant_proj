"""
Main Execution Module for Hypothesis Testing Project

This module orchestrates the complete data processing and hypothesis validation pipeline
by integrating all the modular components and implementing the logic from mid.py.
"""

# Standard library imports
import warnings
from typing import Dict, Any, Tuple

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# Optional imports
try:
    import japanize_matplotlib
except ImportError:
    print("Warning: japanize_matplotlib not available")

try:
    import shap
except ImportError:
    print("Warning: shap not available")

# Local imports
from data_loading import DataLoader
from data_analysis import DataMapper, DataProcessor, DataMerger
from hypothesis_testing import Hypothesis1Validator, StatisticalTests, PerformanceAnalyzer
from config import validate_data_paths, get_config_summary

# Configure warnings
warnings.filterwarnings("ignore")


class HypothesisTestingPipeline:
    """
    Main pipeline class for the hypothesis testing project.
    
    This class orchestrates the entire data processing and analysis workflow
    based on the logic from mid.py.
    """
    
    def __init__(self):
        """Initialize the pipeline with all necessary components."""
        self.data_loader = DataLoader()
        self.data_mapper = DataMapper()
        self.data_processor = DataProcessor()
        self.data_merger = DataMerger()
        self.hypothesis_validator = Hypothesis1Validator()
        self.statistical_tests = StatisticalTests()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Storage for processed data
        self.processed_data = {}
        self.final_dataset = None
        self.mappings = {}
        self.office_mappings = {}
        
        print("Hypothesis Testing Pipeline initialized successfully!")
    
    def validate_environment(self) -> bool:
        """
        Validate that all required data files exist and environment is ready.
        
        Returns:
            True if environment is valid, False otherwise
        """
        print("Validating environment...")
        
        # Validate data paths
        path_validation = validate_data_paths()
        missing_files = [key for key, exists in path_validation.items() if not exists]
        
        if missing_files:
            print(f"Warning: Missing data files: {missing_files}")
            return False
        
        print("Environment validation successful!")
        return True
    
    def load_and_process_main_data(self) -> Dict[str, Any]:
        """
        Load and process the main datasets following the mid.py logic.
        
        Returns:
            Dictionary containing processed datasets and mappings
        """
        print("Loading and processing main datasets...")
        
        # Load main datasets
        lp_history_df, salary_df, performance_df, punishment_df, president_cup_df = self.data_loader.load_main_datasets()
        
        # Apply LP mappings (from mid.py logic)
        lp_history_processed, lp_mappings = self.data_mapper.apply_lp_mappings(lp_history_df)
        
        # Apply office mappings
        lp_history_processed, office_mappings = self.data_mapper.apply_office_mappings(lp_history_processed)
        
        # Store mappings for later use
        self.mappings = lp_mappings
        self.office_mappings = office_mappings
        
        # Apply mappings to other datasets
        salary_df['LP'] = salary_df['SYAIN_CODE'].map(lp_mappings)
        performance_df['LP'] = performance_df['LP'].map(lp_mappings)
        punishment_df['LP'] = punishment_df['LP NO'].map(lp_mappings)
        president_cup_df['LP'] = president_cup_df['LP'].map(lp_mappings)
        
        # Process dates and clean data (following mid.py logic)
        lp_history_cleaned = self.data_processor.clean_dataframe(
            lp_history_processed, 
            drop_columns=['UNIT']
        )
        
        # Apply year offset
        lp_history_cleaned = self.data_processor.process_date_columns(
            lp_history_cleaned, 
            {'year': 'JOB_YYY'}
        )
        
        salary_df = self.data_processor.process_date_columns(
            salary_df, 
            {'year': 'S_YR'}
        )
        
        president_cup_df = self.data_processor.process_date_columns(
            president_cup_df, 
            {'year': 'CONTEST_YYY'}
        )
        
        # Store processed data
        self.processed_data = {
            'lp_history': lp_history_cleaned,
            'salary': salary_df,
            'performance': performance_df,
            'punishment': punishment_df,
            'president_cup': president_cup_df
        }
        
        print("Main data processing completed!")
        return self.processed_data
    
    def merge_all_datasets(self) -> pd.DataFrame:
        """
        Merge all processed datasets into a final dataset following mid.py logic.
        
        Returns:
            Final merged DataFrame
        """
        print("Merging all datasets...")
        
        if not self.processed_data:
            raise ValueError("Main data must be processed first. Call load_and_process_main_data()")
        
        # Start with LP history and salary merge (following mid.py)
        lp_history_df = self.processed_data['lp_history']
        salary_df = self.processed_data['salary']
        
        # Rename columns for consistency
        lp_history_renamed = lp_history_df.rename(columns={'JOB_YYY': 'S_YR', 'JOB_MM': 'S_MO'})
        
        # Merge LP history with salary
        merged_df = self.data_merger.merge_lp_history_with_salary(lp_history_renamed, salary_df)
        
        # Create date column
        merged_df = self.data_processor.create_date_column(
            merged_df, 'S_YR', 'S_MO', 'JOB_DD', 'Date'
        )
        
        # Clean merged data
        merged_df = self.data_processor.clean_dataframe(
            merged_df, 
            drop_columns=['SYAIN_CODE'],
            drop_na=True
        )
        
        # Process and merge performance data
        performance_df = self.processed_data['performance'].copy()
        performance_df = self.data_processor.process_datetime_column(
            performance_df, 'ym', '%Y%m'
        )
        performance_df = self.data_processor.extract_year_month(performance_df, 'ym')
        performance_df = performance_df.sort_values(by=['LP', 'year', 'month'])
        performance_df = self.data_processor.clean_dataframe(performance_df, drop_columns=['ym'])
        
        # Merge with performance data
        merged_df = self.data_merger.merge_with_performance_data(merged_df, performance_df)
        merged_df = self.data_processor.clean_dataframe(merged_df, drop_columns=['year', 'month'])
        merged_df = merged_df.fillna(0)
        
        # Process and merge punishment data
        punishment_df = self.processed_data['punishment'].copy()
        punishment_df_sub = punishment_df[['LP', 'FUKABI', 'SHOBUN']].copy()
        punishment_df_sub = self.data_processor.process_datetime_column(
            punishment_df_sub, 'FUKABI', '%Y/%m/%d'
        )
        punishment_df_sub = self.data_processor.extract_year_month(punishment_df_sub, 'FUKABI')
        
        # Merge with punishment data
        merged_df = self.data_merger.merge_with_punishment_data(merged_df, punishment_df_sub)
        
        # Merge with president cup data
        president_cup_df = self.processed_data['president_cup']
        merged_df = self.data_merger.merge_with_president_cup_data(merged_df, president_cup_df)
        
        # Add fiscal half column (from mid.py)
        merged_df = self.data_processor.add_fiscal_half_column(merged_df)
        
        self.final_dataset = merged_df
        print(f"Final dataset created with {len(merged_df)} rows and {len(merged_df.columns)} columns")
        
        return merged_df
    
    def run_hypothesis_validation(self) -> Dict[str, Any]:
        """
        Run hypothesis validation using the final dataset.
        
        Returns:
            Dictionary containing validation results
        """
        print("Running hypothesis validation...")
        
        if self.final_dataset is None:
            raise ValueError("Final dataset must be created first. Call merge_all_datasets()")
        
        # Validate Hypothesis 1 (from mid.py)
        h1_results = self.hypothesis_validator.validate_hypothesis_1_performance_prediction(
            self.final_dataset
        )
        
        # Generate comprehensive report
        validation_report = self.hypothesis_validator.create_hypothesis_validation_report(
            'H1_performance_prediction'
        )
        
        print("Hypothesis validation completed!")
        return validation_report
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary report of the entire pipeline.
        
        Returns:
            Dictionary containing the summary report
        """
        summary = {
            'pipeline_info': {
                'total_datasets_loaded': len(self.processed_data),
                'final_dataset_shape': self.final_dataset.shape if self.final_dataset is not None else None,
                'mappings_created': len(self.mappings),
                'office_mappings_created': len(self.office_mappings)
            },
            'data_summary': self.data_loader.get_data_summary(),
            'merge_summary': self.data_merger.get_merge_summary(),
            'hypothesis_summary': self.hypothesis_validator.get_hypothesis_summary(),
            'validation_results': self.hypothesis_validator.get_validation_results()
        }
        
        return summary


def main():
    """
    Main execution function for the hypothesis testing pipeline.
    """
    print("Starting Hypothesis Testing Pipeline...")
    
    # Initialize pipeline
    pipeline = HypothesisTestingPipeline()
    
    # Validate environment
    if not pipeline.validate_environment():
        print("Environment validation failed. Please check data files.")
        return None
    
    # Load and process main data
    processed_data = pipeline.load_and_process_main_data()
    
    # Merge all datasets
    final_dataset = pipeline.merge_all_datasets()
    
    # Run hypothesis validation
    validation_results = pipeline.run_hypothesis_validation()
    
    # Generate summary report
    summary_report = pipeline.generate_summary_report()
    
    # Print summary
    print(f"\n{'='*50}")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Final dataset shape: {final_dataset.shape}")
    print(f"Total mappings created: {len(pipeline.mappings)}")
    print(f"Hypotheses validated: {len(pipeline.hypothesis_validator.validation_results)}")
    print(f"{'='*50}")
    
    return pipeline, final_dataset, validation_results, summary_report


if __name__ == "__main__":
    pipeline, final_dataset, validation_results, summary_report = main()
