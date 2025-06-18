"""
Hypothesis 3 Validator

This module implements the validation logic for Hypothesis 3:
"AMGR historical disciplinary action predicts LP disciplinary action under their management"

Hypothesis: if "(变数1) == 1" then "(变数2) > 0"
- 变数1: AMGRのLP時代の懲戒処分の有無(0:処分なし、1:処分あり)
- 变数2: このAMGRが管理するLPの懲戒処分の有無
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from data_loading.unified_data_loader import get_unified_data_loader


class Hypothesis3Validator:
    """
    Hypothesis 3 validator for AMGR historical disciplinary action analysis.
    
    Tests whether AMGRs who had disciplinary action during their LP period
    are more likely to have LPs under their management who also receive disciplinary action.
    """
    
    def __init__(self):
        """Initialize the hypothesis validator."""
        self.hypotheses = {}
        self.validation_results = {}
        self.data_loader = get_unified_data_loader()
    
    def register_hypothesis(self, hypothesis_id: str, description: str, 
                          null_hypothesis: str, alternative_hypothesis: str) -> None:
        """Register a hypothesis for validation."""
        self.hypotheses[hypothesis_id] = {
            'description': description,
            'null_hypothesis': null_hypothesis,
            'alternative_hypothesis': alternative_hypothesis,
            'registered_at': pd.Timestamp.now()
        }
        print(f"Registered hypothesis: {hypothesis_id}")

    def run_hypothesis_3_with_data_loader(self) -> Dict[str, Any]:
        """
        运行Hypothesis 3，使用统一数据加载器

        Returns:
            验证结果字典
        """
        print("加载数据用于Hypothesis 3验证...")

        # 使用统一数据加载器创建最终数据集
        final_dataset = self.data_loader.create_final_dataset(
            include_lp_history=True,
            include_reward=False,
            include_discipline=True,
            include_performance=False,
            include_mtg=False,
            include_complaints=False,
            include_awards=False,
            include_office_errors=False
        )

        print(f"数据加载完成，数据形状: {final_dataset.shape}")
        print(f"数据列: {list(final_dataset.columns)}")

        # 运行hypothesis验证
        return self.validate_hypothesis_3_amgr_influence(final_dataset)

    def validate_hypothesis_3_amgr_influence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Hypothesis 3: AMGR Historical Disciplinary Action Influence
        
        Hypothesis: if "(变数1) == 1" then "(变数2) > 0"
        - 变数1: AMGRのLP時代の懲戒処分の有無(0:処分なし、1:処分あり)
        - 变数2: このAMGRが管理するLPの懲戒処分の有無
        
        Args:
            df: DataFrame containing the complete merged dataset
            
        Returns:
            Dictionary containing validation results
        """
        print("Validating Hypothesis 3: AMGR Historical Disciplinary Action Influence")
        
        # Register the hypothesis
        self.register_hypothesis(
            "H3_amgr_influence",
            "AMGR Historical Disciplinary Action Influence: AMGRs with past disciplinary action have LPs with more disciplinary issues",
            'Null: AMGR historical disciplinary action does not predict LP disciplinary action',
            'Alternative: AMGR historical disciplinary action predicts LP disciplinary action under their management'
        )
        
        # Apply the analysis logic
        analysis_results = self._analyze_amgr_shobun_influence(df)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(analysis_results)
        
        # Create comparison results
        comparison_results = self._create_amgr_influence_comparison(analysis_results)
        
        # Perform statistical test
        statistical_test = self._perform_hypothesis_3_statistical_test(analysis_results)
        
        # Analyze AMGR patterns
        amgr_patterns = self._analyze_amgr_patterns(analysis_results)
        
        # Compile validation results
        validation_results = {
            'hypothesis_id': 'H3_amgr_influence',
            'data_shape': analysis_results.shape,
            'performance_metrics': performance_metrics,
            'comparison_results': comparison_results,
            'statistical_test': statistical_test,
            'amgr_patterns': amgr_patterns,
            'amgr_influence_analysis': {
                'total_amgrs': analysis_results['AMGR'].nunique(),
                'amgrs_with_history': (analysis_results['AMGR_Has_SHOBUN_History'] == 1).sum(),
                'amgrs_with_lp_violations': (analysis_results['Shobun_Flag'] == 1).sum(),
                'both_conditions_met': ((analysis_results['AMGR_Has_SHOBUN_History'] == 1) & (analysis_results['Shobun_Flag'] == 1)).sum(),
                'prediction_accuracy': self._calculate_prediction_accuracy(analysis_results)
            },
            'conclusion': self._interpret_h3_results(analysis_results),
            'validated_at': pd.Timestamp.now()
        }
        
        # Save results
        self.validation_results['H3_amgr_influence'] = validation_results
        
        print("Hypothesis 3 validation completed")
        return validation_results
    
    def _analyze_amgr_shobun_influence(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze AMGR historical disciplinary action and its influence on LP disciplinary action.
        
        Logic:
        1. Identify AMGRs and their historical disciplinary action when they were LPs
        2. Calculate disciplinary action rates for LPs under each AMGR
        3. Create variables for hypothesis testing
        """
        print("Analyzing AMGR SHOBUN influence...")
        
        # Step 1: Mark AMGRs with SHOBUN history
        amgr_shobun_history_df = self._mark_amgr_shobun_history(dataframe)
        
        # Step 2: Calculate violations under each AMGR
        amgr_lp_violations_df = self._calculate_amgr_lp_violations(dataframe, amgr_shobun_history_df)
        
        # Step 3: Summarize total statistics
        amgr_total_statistics = self._summarize_amgr_total_statistics(amgr_lp_violations_df)
        
        print(f"Analysis completed: {len(amgr_total_statistics)} AMGRs analyzed")
        print(f"AMGRs with SHOBUN history: {(amgr_total_statistics['AMGR_Has_SHOBUN_History'] == 1).sum()}")
        print(f"AMGRs with LP violations: {(amgr_total_statistics['Shobun_Flag'] == 1).sum()}")
        
        return amgr_total_statistics
    
    def _mark_amgr_shobun_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark AMGRs with SHOBUN history when they were LPs."""
        # Handle different RANK column names
        rank_col = 'RANK_x' if 'RANK_x' in df.columns else 'RANK'

        # Get list of all AMGRs (RANK 20) - use AMGR column directly
        amgr_list = df['AMGR'].dropna().unique()

        # Initialize results list
        results = []

        # Use tqdm for progress bar
        for amgr in tqdm(amgr_list, desc="Checking AMGR SHOBUN history"):
            # Check if this person had SHOBUN when they were RANK 10
            had_shobun = df[
                (df['LP'] == amgr) &
                (df[rank_col] == 10) &
                (df['SHOBUN'].notna())
            ].shape[0] > 0

            results.append({
                'AMGR': amgr,
                'Has_SHOBUN_History': int(had_shobun)
            })

        return pd.DataFrame(results)
    
    def _calculate_amgr_lp_violations(self, df: pd.DataFrame, amgr_shobun_history_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate violations for LPs under each AMGR."""
        # Handle different RANK column names
        rank_col = 'RANK_x' if 'RANK_x' in df.columns else 'RANK'

        # Merge AMGR SHOBUN history data
        df = df.merge(amgr_shobun_history_df, left_on='AMGR', right_on='AMGR', how='left')

        amgr_lp_stats_list = []
        amgrs = df['AMGR'].dropna().unique()

        # Use tqdm for progress bar
        for amgr in tqdm(amgrs, desc="Calculating violations for each AMGR"):
            amgr_data = df[df['AMGR'] == amgr]

            for fy in amgr_data['S_YR'].unique():
                # Filter data for current fiscal year
                amgr_fy_data = amgr_data[amgr_data['S_YR'] == fy]
                total_lps = amgr_fy_data[amgr_fy_data[rank_col] == 10]['LP'].nunique()

                # Calculate violations
                shobun_count = amgr_fy_data[amgr_fy_data[rank_col] == 10]['SHOBUN'].notna().sum()

                # Get SHOBUN history
                shobun_history = amgr_fy_data['Has_SHOBUN_History'].fillna(0).max()

                amgr_lp_stats_list.append({
                    'AMGR': amgr,
                    'Fiscal_Year': fy,
                    'Total_LPs': total_lps,
                    'Shobun_Count': shobun_count,
                    'AMGR_Has_SHOBUN_History': shobun_history
                })

        return pd.DataFrame(amgr_lp_stats_list)
    
    def _summarize_amgr_total_statistics(self, amgr_lp_violations_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize total statistics for each AMGR."""
        # Group by AMGR and sum up the relevant columns
        amgr_total_stats = amgr_lp_violations_df.groupby('AMGR').agg({
            'Total_LPs': 'sum',
            'Shobun_Count': 'sum',
            'AMGR_Has_SHOBUN_History': 'first'  # Same for all years
        }).reset_index()
        
        # Rename columns to reflect that these are total values
        amgr_total_stats = amgr_total_stats.rename(columns={
            'Total_LPs': 'Total_LPs_All_Years',
            'Shobun_Count': 'Total_Shobun_Count'
        })
        
        # Add useful ratios
        amgr_total_stats['Shobun_per_LP'] = (
            amgr_total_stats['Total_Shobun_Count'] / 
            amgr_total_stats['Total_LPs_All_Years']
        ).round(3)
        
        # Convert to int
        amgr_total_stats['AMGR_Has_SHOBUN_History'] = amgr_total_stats['AMGR_Has_SHOBUN_History'].astype(int)
        
        # Create flag for Shobun_Count > 0
        amgr_total_stats['Shobun_Flag'] = (amgr_total_stats['Shobun_per_LP'] > 0).astype(int)
        
        return amgr_total_stats

    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis results."""
        if df.empty:
            return {'error': 'No data available'}

        metrics = {}

        for col in ['Total_LPs_All_Years', 'Total_Shobun_Count', 'Shobun_per_LP']:
            if col in df.columns:
                metrics[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'count': len(df[col]),
                    'missing_count': df[col].isna().sum()
                }

        return metrics

    def _create_amgr_influence_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison results for AMGR influence."""
        if df.empty or 'AMGR_Has_SHOBUN_History' not in df.columns:
            return {'error': 'Required variables not found'}

        # Comparison by AMGR_Has_SHOBUN_History
        comparison_amgr = df.groupby('AMGR_Has_SHOBUN_History').agg({
            'Total_LPs_All_Years': ['count', 'mean', 'median', 'std'],
            'Total_Shobun_Count': ['sum', 'mean', 'median', 'std'],
            'Shobun_per_LP': ['mean', 'median', 'std'],
            'Shobun_Flag': ['sum', 'mean']
        }).round(4)

        return {'amgr_influence_comparison': comparison_amgr}

    def _perform_hypothesis_3_statistical_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical test for Hypothesis 3."""
        if df.empty or 'AMGR_Has_SHOBUN_History' not in df.columns or 'Shobun_Flag' not in df.columns:
            return {'error': 'Required variables not found'}

        # Create confusion matrix
        cm = confusion_matrix(df['Shobun_Flag'], df['AMGR_Has_SHOBUN_History'])
        f1 = f1_score(df['Shobun_Flag'], df['AMGR_Has_SHOBUN_History'])

        # Calculate hypothesis-specific metrics
        condition_true = (df['AMGR_Has_SHOBUN_History'] == 1)
        condition_true_count = condition_true.sum()

        # Cases where both condition and outcome are true
        both_true = ((df['AMGR_Has_SHOBUN_History'] == 1) & (df['Shobun_Flag'] == 1)).sum()

        # Hypothesis support rate
        hypothesis_support_rate = both_true / condition_true_count if condition_true_count > 0 else 0

        return {
            'test_type': 'Hypothesis 3 Validation: if AMGR_Has_SHOBUN_History = 1 then Shobun_Flag = 1',
            'confusion_matrix': cm.tolist(),
            'f1_score': f1,
            'condition_true_cases': int(condition_true_count),
            'both_true_cases': int(both_true),
            'hypothesis_support_rate': hypothesis_support_rate,
            'classification_report': classification_report(
                df['Shobun_Flag'], df['AMGR_Has_SHOBUN_History'], output_dict=True
            ),
            'sample_size': len(df)
        }

    def _analyze_amgr_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in AMGR behavior."""
        if df.empty:
            return {'error': 'No data available'}

        patterns = {}

        # AMGR distribution by history
        patterns['amgr_distribution'] = df['AMGR_Has_SHOBUN_History'].value_counts().to_dict()

        # LP violation distribution
        patterns['lp_violation_distribution'] = df['Shobun_Flag'].value_counts().to_dict()

        # Cross-tabulation
        patterns['cross_tabulation'] = pd.crosstab(
            df['AMGR_Has_SHOBUN_History'],
            df['Shobun_Flag'],
            margins=True
        ).to_dict()

        return patterns

    def _calculate_prediction_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate prediction accuracy metrics for Hypothesis 3."""
        if df.empty or 'AMGR_Has_SHOBUN_History' not in df.columns or 'Shobun_Flag' not in df.columns:
            return {'error': 'Required variables not found'}

        # Calculate accuracy metrics
        tp = ((df['AMGR_Has_SHOBUN_History'] == 1) & (df['Shobun_Flag'] == 1)).sum()
        tn = ((df['AMGR_Has_SHOBUN_History'] == 0) & (df['Shobun_Flag'] == 0)).sum()
        fp = ((df['AMGR_Has_SHOBUN_History'] == 1) & (df['Shobun_Flag'] == 0)).sum()
        fn = ((df['AMGR_Has_SHOBUN_History'] == 0) & (df['Shobun_Flag'] == 1)).sum()

        total = len(df)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Hypothesis-specific metrics
        condition_met = (df['AMGR_Has_SHOBUN_History'] == 1).sum()
        outcome_when_condition_met = ((df['AMGR_Has_SHOBUN_History'] == 1) & (df['Shobun_Flag'] == 1)).sum()
        hypothesis_accuracy = outcome_when_condition_met / condition_met if condition_met > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'hypothesis_accuracy': hypothesis_accuracy,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'condition_met_cases': int(condition_met),
            'outcome_when_condition_met': int(outcome_when_condition_met)
        }

    def _interpret_h3_results(self, df: pd.DataFrame) -> str:
        """Interpret the results of Hypothesis 3 validation."""
        if df.empty or 'AMGR_Has_SHOBUN_History' not in df.columns or 'Shobun_Flag' not in df.columns:
            return "Error: Required variables not found for interpretation"

        # Calculate key metrics
        amgrs_with_history = (df['AMGR_Has_SHOBUN_History'] == 1).sum()
        amgrs_with_lp_violations = (df['Shobun_Flag'] == 1).sum()
        both_count = ((df['AMGR_Has_SHOBUN_History'] == 1) & (df['Shobun_Flag'] == 1)).sum()

        accuracy_metrics = self._calculate_prediction_accuracy(df)
        hypothesis_accuracy = accuracy_metrics.get('hypothesis_accuracy', 0)

        f1 = f1_score(df['Shobun_Flag'], df['AMGR_Has_SHOBUN_History'])

        conclusion = f"Hypothesis 3 Results: {amgrs_with_history} AMGRs with disciplinary history, "
        conclusion += f"{amgrs_with_lp_violations} AMGRs with LP violations, {both_count} overlap. "
        conclusion += f"Hypothesis accuracy: {hypothesis_accuracy:.3f}, F1-score: {f1:.3f}"

        return conclusion
