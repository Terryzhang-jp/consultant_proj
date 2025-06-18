"""
Hypothesis 4 Validator

This module implements the validation logic for Hypothesis 4:
"Decision tree prediction using LP award rate and compliance violation rate"

Hypothesis: "变数1,2を利用した決定木による判定" then (变数3) = 1
- 变数1: 在籍年数の半分以上で入賞したLPの割合
- 变数2: LPひとりあたりの平均年間コンプライアンス違反疑件数
- 变数3: 営業所がその後1年間で受ける懲戒処分の有無
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using sklearn DecisionTreeClassifier instead.")
    from sklearn.tree import DecisionTreeClassifier

from ..shared.statistical_tests import StatisticalTests
from ..shared.performance_analyzer import PerformanceAnalyzer


class Hypothesis4Validator:
    """
    Hypothesis 4 validator for decision tree prediction analysis.
    
    Tests whether LP award rates and compliance violation rates can predict
    future disciplinary actions using decision tree models.
    """
    
    def __init__(self):
        """Initialize the hypothesis validator."""
        self.hypotheses = {}
        self.validation_results = {}
        self.statistical_tests = StatisticalTests()
        self.performance_analyzer = PerformanceAnalyzer()
        self.model = None
        self.feature_importance = None
    
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
    
    def validate_hypothesis_4_decision_tree_prediction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Hypothesis 4: Decision Tree Prediction
        
        Hypothesis: "变数1,2を利用した決定木による判定" then (变数3) = 1
        - 变数1: 在籍年数の半分以上で入賞したLPの割合
        - 变数2: LPひとりあたりの平均年間コンプライアンス違反疑件数
        - 变数3: 営業所がその後1年間で受ける懲戒処分の有無
        
        Args:
            df: DataFrame containing the complete merged dataset
            
        Returns:
            Dictionary containing validation results
        """
        print("Validating Hypothesis 4: Decision Tree Prediction")
        
        # Register the hypothesis
        self.register_hypothesis(
            "H4_decision_tree_prediction",
            "Decision Tree Prediction: LP award rates and compliance violations predict future disciplinary action",
            'Null: LP award rates and compliance violations cannot predict future disciplinary action',
            'Alternative: LP award rates and compliance violations can predict future disciplinary action using decision trees'
        )
        
        # Apply the analysis logic
        analysis_results = self._analyze_decision_tree_prediction(df)
        
        # Train and evaluate decision tree model
        model_results = self._train_decision_tree_model(analysis_results)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(analysis_results)
        
        # Create comparison results
        comparison_results = self._create_decision_tree_comparison(analysis_results)
        
        # Analyze feature importance
        feature_importance = self._analyze_feature_importance()
        
        # Compile validation results
        validation_results = {
            'hypothesis_id': 'H4_decision_tree_prediction',
            'data_shape': analysis_results.shape,
            'performance_metrics': performance_metrics,
            'comparison_results': comparison_results,
            'model_results': model_results,
            'feature_importance': feature_importance,
            'decision_tree_analysis': {
                'total_records': len(analysis_results),
                'prediction_accuracy': model_results.get('accuracy', 0),
                'f1_score': model_results.get('f1_score', 0),
                'feature_count': len(feature_importance) if feature_importance else 0
            },
            'conclusion': self._interpret_h4_results(analysis_results, model_results),
            'validated_at': pd.Timestamp.now()
        }
        
        # Save results
        self.validation_results['H4_decision_tree_prediction'] = validation_results
        
        print("Hypothesis 4 validation completed")
        return validation_results
    
    def _analyze_decision_tree_prediction(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze decision tree prediction using LP award rates and compliance violations.
        
        Logic:
        1. Calculate fiscal years and tenure statistics
        2. Calculate LP award rates (Variable 1)
        3. Calculate compliance violation rates (Variable 2)
        4. Predict future disciplinary actions (Variable 3)
        """
        print("Analyzing decision tree prediction...")
        
        # Step 1: Calculate tenure and contest statistics
        tenure_contest_stats = self._calculate_tenure_and_contest_stats(dataframe)
        
        # Step 2: Calculate AMGR qualification rates
        amgr_qualification_rate = self._calculate_amgr_qualification_rate_adjusted(dataframe, tenure_contest_stats)
        
        # Step 3: Calculate AMGR violations
        amgr_violations = self._calculate_amgr_violations(dataframe)
        
        # Step 4: Merge and prepare final analysis data
        final_analysis_df = pd.merge(amgr_qualification_rate, amgr_violations, 
                                   on=['AMGR', 'Fiscal_Year'], how='left')
        
        # Step 5: Prepare data for prediction
        prediction_data = self._prepare_data_for_prediction(final_analysis_df)
        
        print(f"Analysis completed: {len(prediction_data)} prediction records")
        
        return prediction_data

    def _calculate_tenure_and_contest_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate tenure and contest statistics for each LP."""
        print("Calculating tenure and contest statistics...")

        # Calculate fiscal year for each record
        df['Fiscal_Year'] = df['S_YR'] + (df['S_MO'] >= 10).astype(int)

        # Calculate tenure statistics
        tenure_stats = df.groupby('LP').agg({
            'Fiscal_Year': ['min', 'max'],
            'コンプライアンス': 'sum'  # Compliance violations
        }).reset_index()

        tenure_stats.columns = ['LP', 'Start_Fiscal_Year', 'End_Fiscal_Year', 'Compliance_Count']
        tenure_stats['Tenure_Years'] = tenure_stats['End_Fiscal_Year'] - tenure_stats['Start_Fiscal_Year'] + 1

        # Calculate contest years
        contest_data = df[df['CONTEST_ID'].notna()] if 'CONTEST_ID' in df.columns else pd.DataFrame()
        if not contest_data.empty:
            contest_years = contest_data.groupby('LP')['Fiscal_Year'].nunique().reset_index()
            contest_years.columns = ['LP', 'Contest_Years']
        else:
            contest_years = pd.DataFrame({'LP': tenure_stats['LP'], 'Contest_Years': 0})

        # Merge and calculate qualification
        merged_df = pd.merge(tenure_stats, contest_years, on='LP', how='left')
        merged_df['Contest_Years'] = merged_df['Contest_Years'].fillna(0)

        # Mark qualified LPs (contest years > half of tenure years AND no compliance issues)
        merged_df['Qualified'] = ((merged_df['Contest_Years'] > (merged_df['Tenure_Years'] / 2)) &
                                 (merged_df['Compliance_Count'] == 0)).astype(int)

        return merged_df

    def _calculate_amgr_qualification_rate_adjusted(self, df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate AMGR qualification rates."""
        print("Calculating AMGR qualification rates...")

        amgr_stats_list = []

        # Get AMGRs (assuming RANK=20 or similar identifier)
        if 'RANK' in df.columns:
            amgrs = df[df['RANK'] == 20]['LP'].unique()
        elif 'AMGR' in df.columns:
            amgrs = df['AMGR'].unique()
        else:
            print("Warning: No AMGR identifier found")
            return pd.DataFrame()

        # Use tqdm for progress tracking
        for amgr in tqdm(amgrs, desc="Processing AMGRs"):
            amgr_data = df[df['AMGR'] == amgr] if 'AMGR' in df.columns else df[df['LP'] == amgr]

            if 'Fiscal_Year' not in amgr_data.columns:
                amgr_data['Fiscal_Year'] = amgr_data['S_YR'] + (amgr_data['S_MO'] >= 10).astype(int)

            fiscal_years = amgr_data['Fiscal_Year'].unique()

            for fy in fiscal_years:
                # Filter data up to current fiscal year
                amgr_fy_data = amgr_data[amgr_data['Fiscal_Year'] <= fy]
                total_employees = amgr_fy_data['LP'].nunique()

                if total_employees > 0:
                    qualified_employees = merged_df[
                        (merged_df['LP'].isin(amgr_fy_data['LP'])) &
                        (merged_df['Qualified'] == 1)
                    ].shape[0]

                    qualification_rate = qualified_employees / total_employees

                    amgr_stats_list.append({
                        'AMGR': amgr,
                        'Fiscal_Year': fy,
                        'Total_Employees': total_employees,
                        'Qualified_Employees': qualified_employees,
                        'Qualification_Rate': qualification_rate
                    })

        return pd.DataFrame(amgr_stats_list)

    def _calculate_amgr_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate AMGR violations."""
        print("Calculating AMGR violations...")

        if 'Fiscal_Year' not in df.columns:
            df['Fiscal_Year'] = df['S_YR'] + (df['S_MO'] >= 10).astype(int)

        # Group by AMGR and fiscal year
        amgr_violations = df.groupby(['AMGR', 'Fiscal_Year']).agg({
            'コンプライアンス': 'sum',  # Sum compliance violations
            'SHOBUN': 'count'  # Count disciplinary actions
        }).reset_index()

        amgr_violations.columns = ['AMGR', 'Fiscal_Year', 'Compliance_Violations', 'Shobun_Count']

        # Calculate rates
        employee_counts = df.groupby(['AMGR', 'Fiscal_Year'])['LP'].nunique().reset_index()
        employee_counts.columns = ['AMGR', 'Fiscal_Year', 'Employee_Count']

        amgr_violations = pd.merge(amgr_violations, employee_counts, on=['AMGR', 'Fiscal_Year'], how='left')
        amgr_violations['Compliance_Violations_Rate'] = (
            amgr_violations['Compliance_Violations'] / amgr_violations['Employee_Count']
        ).fillna(0)

        return amgr_violations

    def _prepare_data_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction model."""
        print("Preparing data for prediction...")

        data = []

        for amgr in df['AMGR'].unique():
            amgr_data = df[df['AMGR'] == amgr].sort_values('Fiscal_Year')

            for i in range(len(amgr_data) - 1):
                current_year = amgr_data.iloc[i]
                next_year = amgr_data.iloc[i + 1]

                data.append({
                    'AMGR': amgr,
                    'Fiscal_Year': current_year['Fiscal_Year'],
                    'Qualification_Rate': current_year['Qualification_Rate'],
                    'Compliance_Violations_Rate': current_year['Compliance_Violations_Rate'],
                    'Has_Shobun_Next_Year': 1 if next_year['Shobun_Count'] > 0 else 0
                })

        return pd.DataFrame(data)

    def _train_decision_tree_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train decision tree model for prediction."""
        print("Training decision tree model...")

        if df.empty or len(df) < 10:
            return {'error': 'Insufficient data for model training'}

        # Prepare features and target
        feature_columns = ['Qualification_Rate', 'Compliance_Violations_Rate']
        X = df[feature_columns]
        y = df['Has_Shobun_Next_Year']

        if len(y.unique()) < 2:
            return {'error': 'Target variable has only one class'}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        if XGBOOST_AVAILABLE:
            # Calculate class weights for imbalanced data
            scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1

            self.model = XGBClassifier(
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                max_depth=3,
                n_estimators=100
            )
        else:
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(
                random_state=42,
                max_depth=3,
                class_weight='balanced'
            )

        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))

        return {
            'model_type': 'XGBoost' if XGBOOST_AVAILABLE else 'DecisionTree',
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'feature_importance': self.feature_importance,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }

    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis results."""
        if df.empty:
            return {'error': 'No data available'}

        metrics = {}

        for col in ['Qualification_Rate', 'Compliance_Violations_Rate']:
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

    def _create_decision_tree_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison results for decision tree analysis."""
        if df.empty or 'Has_Shobun_Next_Year' not in df.columns:
            return {'error': 'Required variables not found'}

        # Comparison by future disciplinary action
        comparison_shobun = df.groupby('Has_Shobun_Next_Year').agg({
            'Qualification_Rate': ['count', 'mean', 'median', 'std'],
            'Compliance_Violations_Rate': ['mean', 'median', 'std']
        }).round(4)

        return {'decision_tree_comparison': comparison_shobun}

    def _analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze feature importance from the trained model."""
        if self.feature_importance is None:
            return {}

        return self.feature_importance

    def _interpret_h4_results(self, df: pd.DataFrame, model_results: Dict[str, Any]) -> str:
        """Interpret the results of Hypothesis 4 validation."""
        if df.empty:
            return "Error: No data available for interpretation"

        # Calculate key metrics
        total_records = len(df)
        future_discipline_count = (df['Has_Shobun_Next_Year'] == 1).sum()

        accuracy = model_results.get('accuracy', 0)
        f1 = model_results.get('f1_score', 0)
        model_type = model_results.get('model_type', 'Unknown')

        conclusion = f"Hypothesis 4 Results: {total_records} prediction records, "
        conclusion += f"{future_discipline_count} with future disciplinary action. "
        conclusion += f"{model_type} model accuracy: {accuracy:.3f}, F1-score: {f1:.3f}"

        return conclusion
