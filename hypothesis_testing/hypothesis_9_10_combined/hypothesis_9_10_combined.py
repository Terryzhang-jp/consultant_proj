"""
Hypothesis 9+10 Combined Validator

This module validates the combined hypothesis:
"When both H9 and H10 conditions are met, disciplinary actions occur in the following year."

Combined Logic: (H9_condition AND H10_condition) → Next_Year_Disciplinary_Actions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class Hypothesis9And10CombinedValidator:
    """
    Validates the combined Hypothesis 9+10:
    
    H9 (变数2): Low SASHIHIKI (< 200,000) for 6+ consecutive months + LP leaves within 2 years
    H10 (变数1): AMGR changes 3+ times within 1 year
    Combined: If (H9 AND H10) then disciplinary actions in next year
    """
    
    def __init__(self):
        self.results = None
        self.analysis_summary = {}
    
    def analyze_hypothesis_9_condition(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze Hypothesis 9 condition: Low SASHIHIKI + LP turnover within 2 years
        
        Returns DataFrame with H9 condition results by office and year
        """
        print("🔍 分析 Hypothesis 9 条件：低薪酬连续6个月 + 2年内离职...")
        
        # Group by office and year
        office_groups = dataframe.groupby(['OFFICE', 'S_YR'])
        h9_results = []
        
        for (office, year), group in office_groups:
            # Focus on LPs only
            lp_group = group[group['RANK_x'] == 10]
            
            if len(lp_group) == 0:
                continue
            
            total_lps = lp_group['LP'].nunique()
            problem_lps_count = 0
            
            # Analyze each LP
            for lp in lp_group['LP'].unique():
                lp_data = lp_group[lp_group['LP'] == lp].sort_values('Date')
                
                # Check for 6+ consecutive months of low SASHIHIKI
                sashihiki_series = lp_data['SASHIHIKI']
                low_sashihiki_streak = False
                
                if len(sashihiki_series) >= 6:
                    # Rolling window to check consecutive low performance
                    low_performance_rolling = (sashihiki_series < 200000).rolling(
                        window=6, min_periods=6
                    ).sum()
                    low_sashihiki_streak = (low_performance_rolling >= 6).any()
                
                # Check if LP left within 2 years
                start_date = lp_data['Date'].min()
                end_date = lp_data['Date'].max()
                duration_days = (end_date - start_date).days
                has_left = 'T' in lp_data['STATUS'].values
                left_within_2years = has_left and duration_days <= 730
                
                # H9 condition: Both low performance AND left within 2 years
                if low_sashihiki_streak and left_within_2years:
                    problem_lps_count += 1
            
            h9_results.append({
                'OFFICE': office,
                'YEAR': year,
                'TOTAL_LPS': total_lps,
                'H9_PROBLEM_LPS_COUNT': problem_lps_count,
                'H9_CONDITION_MET': problem_lps_count > 0  # H9 condition
            })
        
        return pd.DataFrame(h9_results)
    
    def analyze_hypothesis_10_condition(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze Hypothesis 10 condition: AMGR changes 3+ times within 1 year
        
        Returns DataFrame with H10 condition results by office and year
        """
        print("🔍 分析 Hypothesis 10 条件：AMGR 1年内变更3次以上...")
        
        # Group by office and year
        office_groups = dataframe.groupby(['OFFICE', 'S_YR'])
        h10_results = []
        
        # Track previous year AMGRs for each office
        office_amgr_history = {}
        
        for (office, year), group in office_groups:
            # Get current year AMGRs
            current_amgrs = set(group['AMGR'].unique())
            
            # Get previous year AMGRs
            prev_amgrs = office_amgr_history.get(office, set())
            
            # Calculate AMGR changes (new + departed)
            new_amgrs = current_amgrs - prev_amgrs
            departed_amgrs = prev_amgrs - current_amgrs
            total_changes = len(new_amgrs) + len(departed_amgrs)
            
            # H10 condition: 3+ AMGR changes
            h10_condition_met = total_changes >= 3
            
            h10_results.append({
                'OFFICE': office,
                'YEAR': year,
                'CURRENT_AMGRS': list(current_amgrs),
                'NEW_AMGRS': list(new_amgrs),
                'DEPARTED_AMGRS': list(departed_amgrs),
                'TOTAL_AMGR_CHANGES': total_changes,
                'H10_CONDITION_MET': h10_condition_met  # H10 condition
            })
            
            # Update history
            office_amgr_history[office] = current_amgrs
        
        return pd.DataFrame(h10_results)
    
    def get_next_year_disciplinary_actions(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Get disciplinary actions (SHOBUN) for the next year by office
        """
        print("🔍 获取次年纪律处分数据...")
        
        # Group by office and year to count SHOBUN
        shobun_by_office_year = dataframe.groupby(['OFFICE', 'S_YR']).agg({
            'SHOBUN': lambda x: x.notna().sum()
        }).reset_index()
        
        shobun_by_office_year.columns = ['OFFICE', 'YEAR', 'SHOBUN_COUNT']
        
        # Shift to get next year's SHOBUN count
        shobun_by_office_year['NEXT_YEAR_SHOBUN_COUNT'] = (
            shobun_by_office_year.groupby('OFFICE')['SHOBUN_COUNT'].shift(-1)
        )
        
        # Create binary flag for next year disciplinary actions
        shobun_by_office_year['NEXT_YEAR_DISCIPLINARY_ACTION'] = (
            shobun_by_office_year['NEXT_YEAR_SHOBUN_COUNT'] > 0
        ).astype(int)
        
        # Remove rows without next year data
        result = shobun_by_office_year.dropna(subset=['NEXT_YEAR_SHOBUN_COUNT'])
        
        return result[['OFFICE', 'YEAR', 'NEXT_YEAR_SHOBUN_COUNT', 'NEXT_YEAR_DISCIPLINARY_ACTION']]

    def validate_combined_hypothesis(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the combined Hypothesis 9+10:
        If (H9_condition AND H10_condition) then Next_Year_Disciplinary_Actions

        Returns comprehensive analysis results
        """
        print("🚀 开始验证组合假设 9+10...")

        # Step 1: Analyze H9 condition
        h9_results = self.analyze_hypothesis_9_condition(dataframe)

        # Step 2: Analyze H10 condition
        h10_results = self.analyze_hypothesis_10_condition(dataframe)

        # Step 3: Get next year disciplinary actions
        disciplinary_results = self.get_next_year_disciplinary_actions(dataframe)

        # Step 4: Merge all results
        print("🔗 合并分析结果...")
        combined_results = h9_results.merge(
            h10_results, on=['OFFICE', 'YEAR'], how='inner'
        ).merge(
            disciplinary_results, on=['OFFICE', 'YEAR'], how='inner'
        )

        # Step 5: Create combined condition (H9 AND H10)
        combined_results['H9_AND_H10_CONDITION'] = (
            combined_results['H9_CONDITION_MET'] &
            combined_results['H10_CONDITION_MET']
        ).astype(int)

        # Store results
        self.results = combined_results

        # Step 6: Calculate validation metrics
        validation_results = self._calculate_validation_metrics(combined_results)

        return validation_results

    def _calculate_validation_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive validation metrics"""
        from sklearn.metrics import confusion_matrix, classification_report, f1_score

        print("📊 计算验证指标...")

        # Extract variables for analysis
        X = results_df['H9_AND_H10_CONDITION']  # Combined condition
        y = results_df['NEXT_YEAR_DISCIPLINARY_ACTION']  # Outcome

        # Basic statistics
        total_cases = len(results_df)
        h9_cases = results_df['H9_CONDITION_MET'].sum()
        h10_cases = results_df['H10_CONDITION_MET'].sum()
        combined_cases = results_df['H9_AND_H10_CONDITION'].sum()
        disciplinary_cases = results_df['NEXT_YEAR_DISCIPLINARY_ACTION'].sum()

        # Confusion matrix
        conf_matrix = confusion_matrix(y, X)
        tn, fp, fn, tp = conf_matrix.ravel()

        # Classification metrics
        class_report = classification_report(y, X, output_dict=True)
        f1 = f1_score(y, X)

        # Store analysis summary
        self.analysis_summary = {
            'total_cases': total_cases,
            'h9_condition_cases': h9_cases,
            'h10_condition_cases': h10_cases,
            'combined_condition_cases': combined_cases,
            'disciplinary_action_cases': disciplinary_cases,
            'confusion_matrix': conf_matrix,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'f1_score': f1,
            'classification_report': class_report
        }

        return self.analysis_summary

    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        if not self.analysis_summary:
            print("❌ 请先运行 validate_combined_hypothesis() 方法")
            return

        summary = self.analysis_summary

        print("\n" + "="*60)
        print("📋 HYPOTHESIS 9+10 组合验证结果")
        print("="*60)

        print(f"\n📊 基础统计:")
        print(f"   总样本数: {summary['total_cases']}")
        print(f"   H9条件满足: {summary['h9_condition_cases']} ({summary['h9_condition_cases']/summary['total_cases']*100:.1f}%)")
        print(f"   H10条件满足: {summary['h10_condition_cases']} ({summary['h10_condition_cases']/summary['total_cases']*100:.1f}%)")
        print(f"   组合条件满足(H9 AND H10): {summary['combined_condition_cases']} ({summary['combined_condition_cases']/summary['total_cases']*100:.1f}%)")
        print(f"   次年纪律处分: {summary['disciplinary_action_cases']} ({summary['disciplinary_action_cases']/summary['total_cases']*100:.1f}%)")

        print(f"\n🎯 混淆矩阵:")
        print(f"   真阴性(TN): {summary['true_negatives']}")
        print(f"   假阳性(FP): {summary['false_positives']}")
        print(f"   假阴性(FN): {summary['false_negatives']}")
        print(f"   真阳性(TP): {summary['true_positives']}")

        print(f"\n📈 性能指标:")
        accuracy = (summary['true_positives'] + summary['true_negatives']) / summary['total_cases']
        precision = summary['classification_report']['1']['precision']
        recall = summary['classification_report']['1']['recall']

        print(f"   准确率(Accuracy): {accuracy:.3f}")
        print(f"   精确率(Precision): {precision:.3f}")
        print(f"   召回率(Recall): {recall:.3f}")
        print(f"   F1分数: {summary['f1_score']:.3f}")

        print("\n" + "="*60)
