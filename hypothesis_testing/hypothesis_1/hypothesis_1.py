"""
Correct Hypothesis 1 Validator

This module implements the exact logic from mid.py for Hypothesis 1 validation.
Matches the hypothesis definition provided by the user.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

from data_loading.unified_data_loader import get_unified_data_loader


class Hypothesis1ValidatorCorrect:
    """
    Correct implementation of Hypothesis 1 validator that matches mid.py logic exactly.

    Hypothesis: if "(变数1) >= 6" then "(变数2) >= 1"
    - 变数1(t): "LP(合計103461LP)のうち、所属するOFFICEのSASHIHIKIより低い半年単位のブロックが連続した数"
    - 变数2(t): "今後半年間で懲戒処分を受ける数"
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

    def run_hypothesis_1_with_data_loader(self) -> Dict[str, Any]:
        """
        运行Hypothesis 1，使用统一数据加载器

        Returns:
            验证结果字典
        """
        print("加载数据用于Hypothesis 1验证...")

        # 使用统一数据加载器创建最终数据集
        final_dataset = self.data_loader.create_final_dataset(
            include_lp_history=True,
            include_reward=True,
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
        return self.validate_hypothesis_1_performance_prediction(final_dataset)

    def validate_hypothesis_1_performance_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Hypothesis 1: LP Disciplinary Action Prediction

        This implements the EXACT logic from mid.py analyze_lp_sashihiki_with_shobun_optimized function.

        Hypothesis: if "(变数1) >= 6" then "(变数2) >= 1"
        - 变数1(t): LP(合計103461LP)のうち、所属するOFFICEのSASHIHIKIより低い半年単位のブロックが連続した数
        - 变数2(t): 今後半年間で懲戒処分を受ける数

        Args:
            df: DataFrame containing the complete merged dataset

        Returns:
            Dictionary containing validation results
        """
        print("Validating Hypothesis 1: LP Disciplinary Action Prediction (Correct Implementation)")

        # Register the hypothesis with correct description (EXACT from mid.py)
        self.register_hypothesis(
            "H1_performance_prediction",
            "LP Disciplinary Action Prediction: Sustained low SASHIHIKI vs AMGR average predicts future disciplinary action",
            'Null: 6+ consecutive periods below AMGR SASHIHIKI average does not predict disciplinary action',
            'Alternative: 6+ consecutive periods below AMGR SASHIHIKI average predicts disciplinary action in next period'
        )
        
        # Apply the EXACT logic from mid.py
        analysis_results = self._analyze_lp_sashihiki_with_shobun_optimized(df)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(analysis_results)
        
        # Create comparison results
        comparison_results = self._create_sustained_low_comparison(analysis_results)
        
        # Perform statistical test (confusion matrix analysis)
        statistical_test = self._perform_hypothesis_1_statistical_test(analysis_results)
        
        # Analyze trends over fiscal periods
        trend_results = self._analyze_fiscal_trends(analysis_results)
        
        # Compile validation results
        validation_results = {
            'hypothesis_id': 'H1_performance_prediction',
            'data_shape': analysis_results.shape,
            'performance_metrics': performance_metrics,
            'comparison_results': comparison_results,
            'statistical_test': statistical_test,
            'trend_analysis': trend_results,
            'sustained_low_analysis': {
                'total_lps': analysis_results['LP'].nunique(),
                'sustained_low_count': (analysis_results['Sustained_Low'] == 1).sum(),
                'shobun_next_half_count': (analysis_results['SHOBUN_in_Next_Half'] == 1).sum(),
                'both_true_count': ((analysis_results['Sustained_Low'] == 1) & (analysis_results['SHOBUN_in_Next_Half'] == 1)).sum(),
                'prediction_accuracy': self._calculate_prediction_accuracy(analysis_results)
            },
            'conclusion': self._interpret_h1_results_correct(analysis_results),
            'validated_at': pd.Timestamp.now()
        }
        
        # Save results
        self.validation_results['H1_performance_prediction'] = validation_results
        
        print("Hypothesis 1 validation completed")
        return validation_results
    
    def _analyze_lp_sashihiki_with_shobun_optimized(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        EXACT implementation of the analyze_lp_sashihiki_with_shobun_optimized function from mid.py

        Updated for correct Hypothesis 1:
        - 变数1: LP所属するOFFICEのSASHIHIKIより低い半年単位のブロックが連続した数
        - 变数2: 今後半年間で懲戒処分を受ける数
        """
        print("Applying exact mid.py logic for LP SASHIHIKI analysis...")

        # Step 1: Create FISCAL_HALF for the ENTIRE dataframe first (EXACT from original)
        dataframe_with_fiscal = dataframe.copy()
        if 'FISCAL_HALF' not in dataframe_with_fiscal.columns:
            def get_fiscal_half_vectorized(s_yr, s_mo):
                """Exact replica of original get_fiscal_half function"""
                conditions = [
                    (s_mo >= 4) & (s_mo <= 9),  # First half
                    s_mo < 4,                   # Second half of previous year
                ]
                choices = [
                    s_yr.astype(str) + '_H1',           # First half
                    (s_yr - 1).astype(str) + '_H2',     # Second half of previous year
                ]
                return np.select(conditions, choices, default=s_yr.astype(str) + '_H2')  # Default: second half

            dataframe_with_fiscal['FISCAL_HALF'] = get_fiscal_half_vectorized(
                dataframe_with_fiscal['S_YR'], dataframe_with_fiscal['S_MO']
            )

        # Step 2: Filter necessary columns and rank 10 data (EXACT from mid.py)
        base_needed_cols = ['LP', 'RANK_x', 'S_YR', 'S_MO', 'SASHIHIKI', 'AMGR', 'SHOBUN', 'FISCAL_HALF']

        # Handle different RANK column names
        rank_col = 'RANK_x' if 'RANK_x' in dataframe_with_fiscal.columns else 'RANK'
        if rank_col in dataframe_with_fiscal.columns:
            # Filter for RANK_x == 10 ONLY (EXACT from mid.py)
            available_cols = [col for col in base_needed_cols if col in dataframe_with_fiscal.columns]
            available_cols = [col if col != 'RANK_x' else rank_col for col in available_cols]
            lp_df = dataframe_with_fiscal[dataframe_with_fiscal[rank_col] == 10][available_cols].copy()
        else:
            print("Warning: No RANK column found, using all data")
            available_cols = [col for col in base_needed_cols if col in dataframe_with_fiscal.columns]
            lp_df = dataframe_with_fiscal[available_cols].copy()

        if len(lp_df) == 0:
            print("Warning: No data after filtering, using all data")
            lp_df = dataframe_with_fiscal.copy()

        # Step 3 & 4: Calculate averages by AMGR (EXACT from mid.py)
        # Use LP data for individual averages
        lp_avg_sashihiki = lp_df.groupby(['LP', 'FISCAL_HALF', 'AMGR'])['SASHIHIKI'].mean().reset_index()

        # Use FULL dataframe for AMGR averages (EXACT from original)
        amgr_avg_sashihiki = dataframe_with_fiscal.groupby(['AMGR', 'FISCAL_HALF'])['SASHIHIKI'].mean().reset_index()
        amgr_avg_sashihiki.columns = ['AMGR', 'FISCAL_HALF', 'AMGR_AVG_SASHIHIKI']

        # Merge operations (EXACT from mid.py)
        result_df = pd.merge(lp_avg_sashihiki, amgr_avg_sashihiki, on=['AMGR', 'FISCAL_HALF'], how='left')

        # Step 5: Vectorized comparison operations (EXACT from mid.py)
        result_df['Below_AMGR_Avg'] = result_df['SASHIHIKI'] < result_df['AMGR_AVG_SASHIHIKI']

        # Step 6: Rolling window calculation for sustained low performance (EXACT from mid.py)
        result_df = result_df.sort_values(['LP', 'FISCAL_HALF'])
        result_df['Rolling_Low'] = result_df.groupby('LP')['Below_AMGR_Avg'].rolling(
            window=6, min_periods=6).sum().reset_index(0, drop=True)

        # 变数1: Sustained_Low (EXACT from mid.py)
        result_df['Sustained_Low'] = (result_df['Rolling_Low'] >= 6).astype(int)

        # Step 7: Check next half-year SHOBUN (exact from mid.py)
        shobun_df = lp_df[['LP', 'FISCAL_HALF', 'SHOBUN']].dropna(subset=['SHOBUN'])
        shobun_df['Has_SHOBUN'] = True

        # Create next half-year mapping (exact from mid.py)
        # Sort fiscal halves properly: 1988_H1, 1988_H2, 1989_H1, 1989_H2, etc.
        fiscal_half_order = sorted(result_df['FISCAL_HALF'].unique(),
                                 key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
        next_half_mapping = dict(zip(fiscal_half_order[:-1], fiscal_half_order[1:]))
        result_df['Next_FISCAL_HALF'] = result_df['FISCAL_HALF'].map(next_half_mapping)

        # Merge SHOBUN information (exact from mid.py)
        result_df = pd.merge(
            result_df,
            shobun_df[['LP', 'FISCAL_HALF', 'Has_SHOBUN']],
            left_on=['LP', 'Next_FISCAL_HALF'],
            right_on=['LP', 'FISCAL_HALF'],
            how='left',
            suffixes=('', '_next')
        )

        # 变数2: SHOBUN_in_Next_Half (EXACT from mid.py)
        result_df['SHOBUN_in_Next_Half'] = result_df['Has_SHOBUN'].fillna(False).astype(int)

        # Clean temporary columns (EXACT from mid.py)
        result_df = result_df.drop(
            ['FISCAL_HALF_next', 'Has_SHOBUN', 'Next_FISCAL_HALF', 'Rolling_Low'],
            axis=1, errors='ignore'
        )

        print(f"Analysis completed: {len(result_df)} records, {result_df['LP'].nunique()} unique LPs")
        print(f"Sustained_Low (>= 6 consecutive low AMGR blocks): {(result_df['Sustained_Low'] == 1).sum()} cases")
        print(f"SHOBUN_in_Next_Half (disciplinary action in next period): {(result_df['SHOBUN_in_Next_Half'] == 1).sum()} cases")

        return result_df
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis results."""
        metrics = {}
        
        for col in ['SASHIHIKI', 'AMGR_AVG_SASHIHIKI']:
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
    
    def _create_sustained_low_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison results for sustained low performance (EXACT from mid.py)."""
        if 'Sustained_Low' not in df.columns or 'SHOBUN_in_Next_Half' not in df.columns:
            return {'error': 'Required columns not found'}

        # Comparison by Sustained_Low (EXACT from mid.py)
        comparison_sustained = df.groupby('Sustained_Low').agg({
            'SASHIHIKI': ['count', 'mean', 'median', 'std'],
            'SHOBUN_in_Next_Half': ['sum', 'mean']
        }).round(4)

        # Comparison by SHOBUN_in_Next_Half
        comparison_shobun = df.groupby('SHOBUN_in_Next_Half').agg({
            'SASHIHIKI': ['count', 'mean', 'median', 'std'],
            'Sustained_Low': ['sum', 'mean']
        }).round(4)

        return {
            'sustained_low_comparison': comparison_sustained,
            'shobun_comparison': comparison_shobun
        }
    
    def _perform_hypothesis_1_statistical_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform statistical test using confusion matrix (EXACT from mid.py).
        """
        if 'Sustained_Low' not in df.columns or 'SHOBUN_in_Next_Half' not in df.columns:
            return {'error': 'Required columns not found'}

        # Create confusion matrix (EXACT from mid.py)
        cm = confusion_matrix(df['SHOBUN_in_Next_Half'], df['Sustained_Low'])
        f1 = f1_score(df['SHOBUN_in_Next_Half'], df['Sustained_Low'])

        # Calculate hypothesis-specific metrics
        # Cases where Sustained_Low == 1 (condition is true)
        condition_true = (df['Sustained_Low'] == 1)
        condition_true_count = condition_true.sum()

        # Cases where both condition and outcome are true
        both_true = ((df['Sustained_Low'] == 1) & (df['SHOBUN_in_Next_Half'] == 1)).sum()

        # Hypothesis support rate
        hypothesis_support_rate = both_true / condition_true_count if condition_true_count > 0 else 0

        return {
            'test_type': 'Confusion Matrix Analysis (EXACT from mid.py)',
            'confusion_matrix': cm.tolist(),
            'f1_score': f1,
            'condition_true_cases': int(condition_true_count),
            'both_true_cases': int(both_true),
            'hypothesis_support_rate': hypothesis_support_rate,
            'classification_report': classification_report(
                df['SHOBUN_in_Next_Half'], df['Sustained_Low'], output_dict=True
            ),
            'sample_size': len(df)
        }
    
    def _analyze_fiscal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends over fiscal periods for correct hypothesis."""
        if 'FISCAL_HALF' not in df.columns:
            return {'error': 'FISCAL_HALF column not found'}

        # Use correct variable names (EXACT from mid.py)
        agg_dict = {'SASHIHIKI': 'mean'}
        if 'Sustained_Low' in df.columns:
            agg_dict['Sustained_Low'] = 'mean'
        if 'SHOBUN_in_Next_Half' in df.columns:
            agg_dict['SHOBUN_in_Next_Half'] = 'mean'

        trend_data = df.groupby('FISCAL_HALF').agg(agg_dict).reset_index()

        return {'fiscal_trends': trend_data}
    
    def _calculate_prediction_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate prediction accuracy metrics (EXACT from mid.py)."""
        if 'Sustained_Low' not in df.columns or 'SHOBUN_in_Next_Half' not in df.columns:
            return {'error': 'Required columns not found'}

        # Calculate accuracy metrics (EXACT from mid.py)
        tp = ((df['Sustained_Low'] == 1) & (df['SHOBUN_in_Next_Half'] == 1)).sum()
        tn = ((df['Sustained_Low'] == 0) & (df['SHOBUN_in_Next_Half'] == 0)).sum()
        fp = ((df['Sustained_Low'] == 1) & (df['SHOBUN_in_Next_Half'] == 0)).sum()
        fn = ((df['Sustained_Low'] == 0) & (df['SHOBUN_in_Next_Half'] == 1)).sum()

        total = len(df)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Hypothesis-specific metrics
        condition_met = (df['Sustained_Low'] == 1).sum()
        outcome_when_condition_met = ((df['Sustained_Low'] == 1) & (df['SHOBUN_in_Next_Half'] == 1)).sum()
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
    
    def _interpret_h1_results_correct(self, df: pd.DataFrame) -> str:
        """Interpret the results of Hypothesis 1 validation (EXACT from mid.py)."""
        if 'Sustained_Low' not in df.columns or 'SHOBUN_in_Next_Half' not in df.columns:
            return "Error: Required columns not found for interpretation"

        # Calculate key metrics (EXACT from mid.py)
        sustained_low_count = (df['Sustained_Low'] == 1).sum()
        shobun_count = (df['SHOBUN_in_Next_Half'] == 1).sum()
        both_count = ((df['Sustained_Low'] == 1) & (df['SHOBUN_in_Next_Half'] == 1)).sum()

        accuracy_metrics = self._calculate_prediction_accuracy(df)
        hypothesis_accuracy = accuracy_metrics.get('hypothesis_accuracy', 0)
        f1 = f1_score(df['SHOBUN_in_Next_Half'], df['Sustained_Low'])

        conclusion = f"Hypothesis 1 Results: {sustained_low_count} LPs with sustained low performance (6+ consecutive periods below AMGR average), "
        conclusion += f"{shobun_count} LPs with disciplinary action in next period, {both_count} overlap. "
        conclusion += f"Prediction accuracy: {hypothesis_accuracy:.3f}, F1-score: {f1:.3f}"

        return conclusion
    
    def get_validation_results(self, hypothesis_id: Optional[str] = None) -> Dict[str, Any]:
        """Get validation results."""
        if hypothesis_id is None:
            return self.validation_results.copy()
        return {hypothesis_id: self.validation_results.get(hypothesis_id, {})}
    
    def create_hypothesis_validation_report(self, hypothesis_id: str) -> Dict[str, Any]:
        """Create a comprehensive validation report."""
        if hypothesis_id not in self.validation_results:
            raise ValueError(f"No validation results found for hypothesis {hypothesis_id}")
        
        results = self.validation_results[hypothesis_id]
        hypothesis_info = self.hypotheses.get(hypothesis_id, {})
        
        return {
            'hypothesis_information': hypothesis_info,
            'validation_results': results,
            'summary': {
                'hypothesis_id': hypothesis_id,
                'validation_date': results.get('validated_at'),
                'data_size': results.get('data_shape'),
                'conclusion': results.get('conclusion'),
                'statistical_significance': results.get('statistical_test', {}).get('f1_score', 0) > 0.5
            }
        }
    
    def get_hypothesis_summary(self) -> pd.DataFrame:
        """Get a summary of all registered hypotheses."""
        summary_data = []
        for hyp_id, hyp_info in self.hypotheses.items():
            validation_status = "Validated" if hyp_id in self.validation_results else "Not Validated"
            summary_data.append({
                'hypothesis_id': hyp_id,
                'description': hyp_info['description'],
                'validation_status': validation_status,
                'registered_at': hyp_info['registered_at']
            })
        return pd.DataFrame(summary_data)
