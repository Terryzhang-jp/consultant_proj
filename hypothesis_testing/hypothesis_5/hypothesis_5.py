"""
Hypothesis 5 Validator

This module implements the validation logic for Hypothesis 5:
"AMGR LP experience and subordinate disciplinary action relationship"

Hypothesis: if "(变数1<2)" & then (变数2) =1
- 变数1: AMGRがLPの時代の経験年数 (AMGR's experience years as LP)
- 变数2: AMGR時代の配下のLPの懲戒処分の有無 (Disciplinary action of subordinate LPs under AMGR)
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

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Class balancing disabled.")

from ..shared.statistical_tests import StatisticalTests
from ..shared.performance_analyzer import PerformanceAnalyzer


class Hypothesis5Validator:
    """
    Hypothesis 5 validator for AMGR experience and subordinate disciplinary action analysis.
    
    Tests whether AMGRs with less than 2 years of LP experience are more likely
    to have subordinate LPs with disciplinary actions.
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
    
    def validate_hypothesis_5_amgr_experience_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Hypothesis 5: AMGR Experience Impact on Subordinate Disciplinary Actions
        
        Hypothesis: if "(变数1<2)" & then (变数2) =1
        - 变数1: AMGRがLPの時代の経験年数 (AMGR's experience years as LP)
        - 变数2: AMGR時代の配下のLPの懲戒処分の有無 (Disciplinary action of subordinate LPs under AMGR)
        
        Args:
            df: DataFrame containing the complete merged dataset
            
        Returns:
            Dictionary containing validation results
        """
        print("Validating Hypothesis 5: AMGR Experience Impact on Subordinate Disciplinary Actions")
        
        # Register the hypothesis
        self.register_hypothesis(
            "H5_amgr_experience_impact",
            "AMGR Experience Impact: AMGRs with <2 years LP experience have more subordinate disciplinary actions",
            'Null: AMGR LP experience does not affect subordinate disciplinary actions',
            'Alternative: AMGRs with <2 years LP experience have more subordinate disciplinary actions'
        )
        
        # Apply the analysis logic
        analysis_results = self._analyze_amgr_experience_impact(df)
        
        # Train and evaluate machine learning model
        model_results = self._train_experience_prediction_model(analysis_results)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(analysis_results)
        
        # Create comparison results
        comparison_results = self._create_experience_comparison(analysis_results)
        
        # Analyze feature importance
        feature_importance = self._analyze_feature_importance()
        
        # Test the specific hypothesis condition (<2 years)
        hypothesis_test_results = self._test_hypothesis_condition(analysis_results)
        
        # Compile validation results
        validation_results = {
            'hypothesis_id': 'H5_amgr_experience_impact',
            'data_shape': analysis_results.shape,
            'performance_metrics': performance_metrics,
            'comparison_results': comparison_results,
            'model_results': model_results,
            'feature_importance': feature_importance,
            'hypothesis_test_results': hypothesis_test_results,
            'amgr_experience_analysis': {
                'total_amgrs': len(analysis_results),
                'amgrs_with_less_than_2_years': len(analysis_results[analysis_results['LP_Experience_Before_AMGR'] < 2]),
                'amgrs_with_disciplinary_actions': len(analysis_results[analysis_results['SHOBUN_Flag'] == 1]),
                'prediction_accuracy': model_results.get('accuracy', 0),
                'f1_score': model_results.get('f1_score', 0)
            },
            'conclusion': self._interpret_h5_results(analysis_results, model_results, hypothesis_test_results),
            'validated_at': pd.Timestamp.now()
        }
        
        # Save results
        self.validation_results['H5_amgr_experience_impact'] = validation_results
        
        print("Hypothesis 5 validation completed")
        return validation_results
    
    def _analyze_amgr_experience_impact(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze AMGR experience impact on subordinate disciplinary actions.
        
        Logic:
        1. Calculate LP experience before AMGR promotion for each AMGR
        2. Calculate AMGR tenure and unique LP count
        3. Count compliance and disciplinary issues for each AMGR's subordinates
        4. Create analysis dataset
        """
        print("Analyzing AMGR experience impact...")
        
        # Remove duplicates
        df_clean = dataframe.drop_duplicates(subset=['LP', 'S_YR', 'S_MO'])
        
        # Apply the verification logic
        analysis_results = self._verify_hypothesis_with_tenure_and_unique_lps(df_clean)
        
        # Create SHOBUN flag
        analysis_results['SHOBUN_Flag'] = (analysis_results['SHOBUN_count'] > 0).astype(int)
        
        # Filter out problematic records if any
        analysis_results = analysis_results[analysis_results['AMGR'] != 'LP_14504']
        
        print(f"Analysis completed: {len(analysis_results)} AMGR records")
        
        return analysis_results

    def _calculate_lp_experience_before_amgr(self, employee_records: list) -> float:
        """Calculate LP experience before promotion to AMGR."""
        lp_experience = 0
        for record in employee_records:
            if record['RANK_x'] == 10:  # LP rank
                lp_experience += 1
            elif record['RANK_x'] == 20:  # AMGR rank
                break  # Stop counting once promoted to AMGR
        return lp_experience / 12  # Convert to years

    def _calculate_amgr_tenure_and_unique_employees(self, amgr_records: pd.DataFrame) -> Tuple[float, int]:
        """Calculate AMGR tenure and unique LP count."""
        if len(amgr_records) == 0:
            return 0, 0

        # Calculate tenure in years
        first_record = amgr_records.iloc[0]
        last_record = amgr_records.iloc[-1]
        tenure_years = (last_record['S_YR'] - first_record['S_YR']) + \
                      (last_record['S_MO'] - first_record['S_MO']) / 12

        # Count unique LPs managed
        unique_lp_count = amgr_records['LP'].nunique()

        return tenure_years, unique_lp_count

    def _verify_hypothesis_with_tenure_and_unique_lps(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        """Verify hypothesis with tenure and unique LP calculations."""
        # Group by AMGR
        amgr_groups = sampled_df.groupby('AMGR')
        amgr_results = []

        # Iterate through each AMGR
        for amgr, group in amgr_groups:
            # Filter records for current AMGR to calculate LP experience
            amgr_records = sampled_df[sampled_df['LP'] == amgr].sort_values(['S_YR', 'S_MO'])
            lp_experience_before_promotion = self._calculate_lp_experience_before_amgr(
                amgr_records.to_dict('records')
            )

            # Check if LP experience is valid
            if lp_experience_before_promotion >= 0:
                # Calculate compliance and disciplinary issues
                compliance_issues = group['コンプライアンス'].notna().sum()
                disciplinary_issues = group['SHOBUN'].notna().sum()

                # Calculate AMGR tenure and unique LP count
                amgr_tenure_years, unique_lp_count = self._calculate_amgr_tenure_and_unique_employees(group)

                # Append results
                amgr_results.append({
                    'AMGR': amgr,
                    'LP_Experience_Before_AMGR': lp_experience_before_promotion,
                    'コンプライアンス_count': compliance_issues,
                    'SHOBUN_count': disciplinary_issues,
                    'Total_Issues': compliance_issues + disciplinary_issues,
                    'AMGR_Tenure_Years': amgr_tenure_years,
                    'Unique_LP_Count': unique_lp_count
                })

        return pd.DataFrame(amgr_results)

    def _train_experience_prediction_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train machine learning model for experience prediction."""
        print("Training experience prediction model...")

        if df.empty or len(df) < 10:
            return {'error': 'Insufficient data for model training'}

        # Prepare features and target
        feature_columns = ['LP_Experience_Before_AMGR', 'AMGR_Tenure_Years']
        X = df[feature_columns]
        y = df['SHOBUN_Flag']

        if len(y.unique()) < 2:
            return {'error': 'Target variable has only one class'}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply SMOTE if available
        if SMOTE_AVAILABLE and len(y_train.unique()) > 1:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Train model
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                max_depth=2,
                learning_rate=0.1,
                n_estimators=100,
                objective='binary:logistic',
                random_state=42
            )
        else:
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(
                max_depth=2,
                random_state=42,
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

        for col in ['LP_Experience_Before_AMGR', 'AMGR_Tenure_Years', 'SHOBUN_count', 'コンプライアンス_count']:
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

    def _create_experience_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create comparison results for experience analysis."""
        if df.empty or 'SHOBUN_Flag' not in df.columns:
            return {'error': 'Required variables not found'}

        # Comparison by experience level (<2 years vs >=2 years)
        df['Experience_Category'] = df['LP_Experience_Before_AMGR'].apply(
            lambda x: 'Less_than_2_years' if x < 2 else '2_years_or_more'
        )

        comparison_experience = df.groupby('Experience_Category').agg({
            'SHOBUN_Flag': ['count', 'sum', 'mean'],
            'LP_Experience_Before_AMGR': ['mean', 'median', 'std'],
            'AMGR_Tenure_Years': ['mean', 'median', 'std']
        }).round(4)

        # Comparison by disciplinary action
        comparison_shobun = df.groupby('SHOBUN_Flag').agg({
            'LP_Experience_Before_AMGR': ['count', 'mean', 'median', 'std'],
            'AMGR_Tenure_Years': ['mean', 'median', 'std']
        }).round(4)

        return {
            'experience_comparison': comparison_experience,
            'disciplinary_comparison': comparison_shobun
        }

    def _analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze feature importance from the trained model."""
        if self.feature_importance is None:
            return {}

        return self.feature_importance

    def _test_hypothesis_condition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test the specific hypothesis condition: experience < 2 years."""
        if df.empty:
            return {'error': 'No data available'}

        # Create experience categories
        less_than_2_years = df[df['LP_Experience_Before_AMGR'] < 2]
        two_years_or_more = df[df['LP_Experience_Before_AMGR'] >= 2]

        # Calculate disciplinary action rates
        less_than_2_rate = less_than_2_years['SHOBUN_Flag'].mean() if len(less_than_2_years) > 0 else 0
        two_years_or_more_rate = two_years_or_more['SHOBUN_Flag'].mean() if len(two_years_or_more) > 0 else 0

        # Statistical test
        from scipy.stats import chi2_contingency

        # Create contingency table
        contingency_table = pd.crosstab(
            df['LP_Experience_Before_AMGR'] < 2,
            df['SHOBUN_Flag'],
            margins=False
        )

        # Perform chi-square test if possible
        chi2_stat, p_value = None, None
        if contingency_table.shape == (2, 2) and contingency_table.min().min() >= 5:
            chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)

        return {
            'less_than_2_years_count': len(less_than_2_years),
            'two_years_or_more_count': len(two_years_or_more),
            'less_than_2_years_disciplinary_rate': less_than_2_rate,
            'two_years_or_more_disciplinary_rate': two_years_or_more_rate,
            'hypothesis_supported': less_than_2_rate > two_years_or_more_rate,
            'rate_difference': less_than_2_rate - two_years_or_more_rate,
            'contingency_table': contingency_table.to_dict(),
            'chi2_statistic': chi2_stat,
            'p_value': p_value
        }

    def _interpret_h5_results(self, df: pd.DataFrame, model_results: Dict[str, Any],
                             hypothesis_test_results: Dict[str, Any]) -> str:
        """Interpret the results of Hypothesis 5 validation."""
        if df.empty:
            return "Error: No data available for interpretation"

        # Calculate key metrics
        total_amgrs = len(df)
        less_than_2_years = len(df[df['LP_Experience_Before_AMGR'] < 2])
        with_disciplinary = (df['SHOBUN_Flag'] == 1).sum()

        accuracy = model_results.get('accuracy', 0)
        f1 = model_results.get('f1_score', 0)
        model_type = model_results.get('model_type', 'Unknown')

        hypothesis_supported = hypothesis_test_results.get('hypothesis_supported', False)
        rate_difference = hypothesis_test_results.get('rate_difference', 0)

        conclusion = f"Hypothesis 5 Results: {total_amgrs} AMGRs analyzed, "
        conclusion += f"{less_than_2_years} with <2 years LP experience, "
        conclusion += f"{with_disciplinary} with subordinate disciplinary actions. "
        conclusion += f"{model_type} model accuracy: {accuracy:.3f}, F1-score: {f1:.3f}. "
        conclusion += f"Hypothesis {'supported' if hypothesis_supported else 'not supported'} "
        conclusion += f"(rate difference: {rate_difference:.3f})"

        return conclusion
