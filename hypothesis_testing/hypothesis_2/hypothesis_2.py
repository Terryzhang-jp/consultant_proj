"""
Hypothesis 2 Validator

This module implements the validation logic for Hypothesis 2:
"LP income concentration predicts future disciplinary action"

Hypothesis: if "(变数1) = 1" then "(变数2) >= 1"
- 变数1(t): LP(2011-2023データ)のうち、1年のSashihikiの50%以上を2か月で稼いでいる
- 变数2(t): そのLPが今後1年間で懲戒処分を受ける数
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")

from data_loading.unified_data_loader import get_unified_data_loader


class Hypothesis2Validator:
    """
    Hypothesis 2 validator for LP income concentration analysis.
    
    Tests whether LPs who earn 50%+ of their annual SASHIHIKI in just 2 months
    are more likely to receive disciplinary action in the following year.
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

    def run_hypothesis_2_with_data_loader(self) -> Dict[str, Any]:
        """
        运行Hypothesis 2，使用统一数据加载器

        Returns:
            验证结果字典
        """
        print("加载数据用于Hypothesis 2验证...")

        # 使用统一数据加载器创建最终数据集
        final_dataset = self.data_loader.create_final_dataset(
            include_lp_history=False,
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
        return self.validate_hypothesis_2_income_concentration(final_dataset)

    def validate_hypothesis_2_income_concentration(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Hypothesis 2: LP Income Concentration Prediction
        
        Hypothesis: if "(变数1) = 1" then "(变数2) >= 1"
        - 变数1: LP(2011-2023データ)のうち、1年のSashihikiの50%以上を2か月で稼いでいる
        - 变数2: そのLPが今後1年間で懲戒処分を受ける数
        
        Args:
            df: DataFrame containing the complete merged dataset
            
        Returns:
            Dictionary containing validation results
        """
        print("Validating Hypothesis 2: LP Income Concentration Prediction")
        
        # Register the hypothesis
        self.register_hypothesis(
            "H2_income_concentration",
            "LP Income Concentration Prediction: High income concentration in 2 months predicts future disciplinary action",
            'Null: Income concentration (50%+ in 2 months) does not predict disciplinary action',
            'Alternative: Income concentration (50%+ in 2 months) predicts disciplinary action in next year'
        )
        
        # Apply the analysis logic
        analysis_results = self._analyze_income_concentration_with_discipline(df)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(analysis_results)
        
        # Create comparison results
        comparison_results = self._create_income_concentration_comparison(analysis_results)
        
        # Perform statistical test
        statistical_test = self._perform_hypothesis_2_statistical_test(analysis_results)
        
        # Analyze trends over years
        trend_results = self._analyze_yearly_trends(analysis_results)
        
        # Compile validation results
        validation_results = {
            'hypothesis_id': 'H2_income_concentration',
            'data_shape': analysis_results.shape,
            'performance_metrics': performance_metrics,
            'comparison_results': comparison_results,
            'statistical_test': statistical_test,
            'trend_analysis': trend_results,
            'income_concentration_analysis': {
                'total_lps': analysis_results['LP'].nunique(),
                'high_concentration_count': (analysis_results['Variable_1'] == 1).sum(),
                'future_discipline_count': (analysis_results['Variable_2'] >= 1).sum(),
                'both_conditions_met': ((analysis_results['Variable_1'] == 1) & (analysis_results['Variable_2'] >= 1)).sum(),
                'prediction_accuracy': self._calculate_prediction_accuracy(analysis_results)
            },
            'conclusion': self._interpret_h2_results(analysis_results),
            'validated_at': pd.Timestamp.now()
        }
        
        # Save results
        self.validation_results['H2_income_concentration'] = validation_results
        
        print("Hypothesis 2 validation completed")
        return validation_results
    
    def _analyze_income_concentration_with_discipline(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze LP income concentration and future disciplinary action.
        
        Logic:
        1. Filter data for 2011-2023
        2. Calculate annual SASHIHIKI for each LP
        3. Find top 2 months for each LP-year
        4. Check if top 2 months >= 50% of annual total
        5. Check for disciplinary action in following year
        """
        print("Analyzing LP income concentration with disciplinary action...")
        
        # Step 1: Filter data for 2011-2023 and necessary columns
        needed_cols = ['LP', 'S_YR', 'S_MO', 'SASHIHIKI', 'SHOBUN']
        available_cols = [col for col in needed_cols if col in dataframe.columns]
        
        if 'SASHIHIKI' not in available_cols:
            print("Warning: SASHIHIKI column not found")
            return pd.DataFrame()
        
        # Filter for 2011-2023 data
        df = dataframe[available_cols].copy()
        df = df[(df['S_YR'] >= 2011) & (df['S_YR'] <= 2023)]
        
        if len(df) == 0:
            print("Warning: No data in 2011-2023 range")
            return pd.DataFrame()
        
        print(f"Filtered data: {len(df)} records for 2011-2023")
        
        # Step 2: Calculate annual SASHIHIKI totals and monthly data
        annual_data = []
        
        # Group by LP and year
        for (lp, year), group in df.groupby(['LP', 'S_YR']):
            # Calculate monthly SASHIHIKI
            monthly_sashihiki = group.groupby('S_MO')['SASHIHIKI'].sum().reset_index()
            
            if len(monthly_sashihiki) < 2:
                continue  # Need at least 2 months of data
            
            # Calculate annual total
            annual_total = monthly_sashihiki['SASHIHIKI'].sum()
            
            if annual_total <= 0:
                continue  # Skip if no income
            
            # Find top 2 months
            top_2_months = monthly_sashihiki.nlargest(2, 'SASHIHIKI')
            top_2_total = top_2_months['SASHIHIKI'].sum()
            
            # Calculate concentration ratio
            concentration_ratio = top_2_total / annual_total if annual_total > 0 else 0
            
            # Variable 1: High concentration (50%+ in top 2 months)
            variable_1 = 1 if concentration_ratio >= 0.5 else 0
            
            # Check for disciplinary action in next year
            next_year = year + 1
            next_year_discipline = df[(df['LP'] == lp) & (df['S_YR'] == next_year) & (df['SHOBUN'].notna())]
            
            # Variable 2: Number of disciplinary actions in next year
            variable_2 = len(next_year_discipline)
            
            annual_data.append({
                'LP': lp,
                'Year': year,
                'Annual_SASHIHIKI': annual_total,
                'Top_2_Months_SASHIHIKI': top_2_total,
                'Concentration_Ratio': concentration_ratio,
                'Variable_1': variable_1,  # High concentration flag
                'Variable_2': variable_2,  # Future discipline count
                'Top_2_Months': list(top_2_months['S_MO'].values)
            })
        
        result_df = pd.DataFrame(annual_data)
        
        if len(result_df) == 0:
            print("Warning: No valid annual data generated")
            return pd.DataFrame()
        
        print(f"Analysis completed: {len(result_df)} LP-year records")
        print(f"High concentration cases (Variable_1=1): {(result_df['Variable_1'] == 1).sum()}")
        print(f"Future discipline cases (Variable_2>=1): {(result_df['Variable_2'] >= 1).sum()}")
        
        return result_df
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis results."""
        if df.empty:
            return {'error': 'No data available'}
        
        metrics = {}
        
        for col in ['Annual_SASHIHIKI', 'Concentration_Ratio']:
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
    
    def _create_income_concentration_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison results for income concentration."""
        if df.empty or 'Variable_1' not in df.columns:
            return {'error': 'Required variables not found'}
        
        # Comparison by Variable_1 (high concentration)
        comparison_var1 = df.groupby('Variable_1').agg({
            'Annual_SASHIHIKI': ['count', 'mean', 'median', 'std'],
            'Concentration_Ratio': ['mean', 'median', 'std'],
            'Variable_2': ['sum', 'mean']
        }).round(4)
        
        return {'income_concentration_comparison': comparison_var1}
    
    def _perform_hypothesis_2_statistical_test(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical test for Hypothesis 2."""
        if df.empty or 'Variable_1' not in df.columns or 'Variable_2' not in df.columns:
            return {'error': 'Required variables not found'}
        
        # Convert Variable_2 to binary (>=1 vs 0)
        variable_2_binary = (df['Variable_2'] >= 1).astype(int)
        
        # Create confusion matrix
        cm = confusion_matrix(variable_2_binary, df['Variable_1'])
        f1 = f1_score(variable_2_binary, df['Variable_1'])
        
        # Calculate hypothesis-specific metrics
        condition_true = (df['Variable_1'] == 1)
        condition_true_count = condition_true.sum()
        
        # Cases where both condition and outcome are true
        both_true = ((df['Variable_1'] == 1) & (df['Variable_2'] >= 1)).sum()
        
        # Hypothesis support rate
        hypothesis_support_rate = both_true / condition_true_count if condition_true_count > 0 else 0
        
        return {
            'test_type': 'Hypothesis 2 Validation: if Variable_1 = 1 then Variable_2 >= 1',
            'confusion_matrix': cm.tolist(),
            'f1_score': f1,
            'condition_true_cases': int(condition_true_count),
            'both_true_cases': int(both_true),
            'hypothesis_support_rate': hypothesis_support_rate,
            'classification_report': classification_report(
                variable_2_binary, df['Variable_1'], output_dict=True
            ),
            'sample_size': len(df)
        }
    
    def _analyze_yearly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends over years."""
        if df.empty or 'Year' not in df.columns:
            return {'error': 'Year column not found'}
        
        trend_data = df.groupby('Year').agg({
            'Variable_1': 'mean',
            'Variable_2': 'mean',
            'Concentration_Ratio': 'mean',
            'Annual_SASHIHIKI': 'mean'
        }).reset_index()
        
        return {'yearly_trends': trend_data}
    
    def _calculate_prediction_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate prediction accuracy metrics for Hypothesis 2."""
        if df.empty or 'Variable_1' not in df.columns or 'Variable_2' not in df.columns:
            return {'error': 'Required variables not found'}
        
        # Convert Variable_2 to binary
        variable_2_binary = (df['Variable_2'] >= 1).astype(int)
        
        # Calculate accuracy metrics
        tp = ((df['Variable_1'] == 1) & (variable_2_binary == 1)).sum()
        tn = ((df['Variable_1'] == 0) & (variable_2_binary == 0)).sum()
        fp = ((df['Variable_1'] == 1) & (variable_2_binary == 0)).sum()
        fn = ((df['Variable_1'] == 0) & (variable_2_binary == 1)).sum()
        
        total = len(df)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Hypothesis-specific metrics
        condition_met = (df['Variable_1'] == 1).sum()
        outcome_when_condition_met = ((df['Variable_1'] == 1) & (variable_2_binary == 1)).sum()
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
    
    def _interpret_h2_results(self, df: pd.DataFrame) -> str:
        """Interpret the results of Hypothesis 2 validation."""
        if df.empty or 'Variable_1' not in df.columns or 'Variable_2' not in df.columns:
            return "Error: Required variables not found for interpretation"
        
        # Calculate key metrics
        high_concentration_count = (df['Variable_1'] == 1).sum()
        future_discipline_count = (df['Variable_2'] >= 1).sum()
        both_count = ((df['Variable_1'] == 1) & (df['Variable_2'] >= 1)).sum()
        
        accuracy_metrics = self._calculate_prediction_accuracy(df)
        hypothesis_accuracy = accuracy_metrics.get('hypothesis_accuracy', 0)
        
        # Convert Variable_2 to binary for F1 calculation
        variable_2_binary = (df['Variable_2'] >= 1).astype(int)
        f1 = f1_score(variable_2_binary, df['Variable_1'])
        
        conclusion = f"Hypothesis 2 Results: {high_concentration_count} LPs with high income concentration (50%+ in 2 months), "
        conclusion += f"{future_discipline_count} LPs with future disciplinary action, {both_count} overlap. "
        conclusion += f"Hypothesis accuracy: {hypothesis_accuracy:.3f}, F1-score: {f1:.3f}"
        
        return conclusion
