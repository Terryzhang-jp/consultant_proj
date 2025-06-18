"""
Hypothesis 6 Validator

This module implements the validation logic for Hypothesis 6:
"Multi-dimensional risk factor analysis for disciplinary action prediction"

Hypothesis: if "(变数1, 连续值)" & (变数2, 连续值) & (变数3, 连续值) & (变数4, 连续值), then (变数5)=1
- 变数1: MTG出席率 (Meeting attendance rate)
- 变数2: 苦情率 (Complaint rate)  
- 变数3: コンプライアンス違反疑義 (Compliance violation suspicion)
- 变数4: 事務ミス (Administrative errors)
- 变数5: 懲戒処分が出る (Disciplinary action occurs)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import os
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Feature importance analysis limited.")

from ..shared.statistical_tests import StatisticalTests
from ..shared.performance_analyzer import PerformanceAnalyzer


class Hypothesis6Validator:
    """
    Hypothesis 6 validator for multi-dimensional risk factor analysis.
    
    Tests whether combination of 4 continuous risk factors (MTG attendance, complaints, 
    compliance violations, administrative errors) can predict disciplinary actions.
    """
    
    def __init__(self):
        """Initialize the hypothesis validator."""
        self.hypotheses = {}
        self.validation_results = {}
        self.statistical_tests = StatisticalTests()
        self.performance_analyzer = PerformanceAnalyzer()
        self.model = None
        self.feature_importance = None
        self.shap_values = None
        self.annual_metrics_df = None
    
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
    
    def load_jimu_miss_data(self, file_path: str) -> pd.DataFrame:
        """Load administrative error (事務ミス) data from Excel file."""
        try:
            if not os.path.exists(file_path):
                print(f"Warning: 事務ミス file not found: {file_path}")
                return pd.DataFrame()
            
            # Try different engines to read Excel file
            try:
                jimu_df = pd.read_excel(file_path, engine='openpyxl')
            except:
                try:
                    jimu_df = pd.read_excel(file_path, engine='xlrd')
                except:
                    jimu_df = pd.read_excel(file_path)
            
            print(f"Loaded 事務ミス data: {len(jimu_df)} rows")
            return jimu_df
            
        except Exception as e:
            print(f"Error loading 事務ミス data: {e}")
            return pd.DataFrame()
    
    def calculate_annual_metrics(self, df: pd.DataFrame, jimu_miss_path: str = None) -> pd.DataFrame:
        """
        Calculate annual metrics for each AMGR including all 4 risk factors.
        
        Args:
            df: Main dataset
            jimu_miss_path: Path to administrative error data
            
        Returns:
            DataFrame with annual metrics per AMGR
        """
        print("Calculating annual metrics with 4 risk factors...")
        
        # Load 事務ミス data
        jimu_df = pd.DataFrame()
        if jimu_miss_path:
            jimu_df = self.load_jimu_miss_data(jimu_miss_path)
        
        # Define fiscal year function (April to March)
        def get_fiscal_year(year, month):
            if month >= 4:
                return year
            else:
                return year - 1
        
        # Add fiscal year to main dataset
        df['Fiscal_Year'] = df.apply(lambda row: get_fiscal_year(row['S_YR'], row['S_MO']), axis=1)
        
        # Add fiscal year to jimu_miss data if available
        if not jimu_df.empty:
            # Try to identify date columns in jimu_miss data
            date_cols = [col for col in jimu_df.columns if any(keyword in str(col).lower() 
                        for keyword in ['date', '日', 'ym', 'year', 'month', 'yr', 'mo'])]
            
            if date_cols:
                print(f"Found date columns in 事務ミス data: {date_cols}")
                # Assume first date column contains year/month info
                date_col = date_cols[0]
                
                # Try to extract year and month
                if 'ym' in str(date_col).lower():
                    # Format like YYYYMM
                    jimu_df['S_YR'] = jimu_df[date_col].astype(str).str[:4].astype(int)
                    jimu_df['S_MO'] = jimu_df[date_col].astype(str).str[4:6].astype(int)
                else:
                    # Try to parse as datetime
                    jimu_df['Date_Parsed'] = pd.to_datetime(jimu_df[date_col], errors='coerce')
                    jimu_df['S_YR'] = jimu_df['Date_Parsed'].dt.year
                    jimu_df['S_MO'] = jimu_df['Date_Parsed'].dt.month
                
                jimu_df['Fiscal_Year'] = jimu_df.apply(
                    lambda row: get_fiscal_year(row['S_YR'], row['S_MO']) if pd.notna(row['S_YR']) else None, 
                    axis=1
                )
        
        # Group by AMGR and Fiscal Year
        annual_metrics = []
        
        amgr_groups = df.groupby(['AMGR', 'Fiscal_Year'])
        
        for (amgr, fiscal_year), group in tqdm(amgr_groups, desc="Processing AMGR annual metrics"):
            if pd.isna(fiscal_year):
                continue
                
            # Calculate metrics for this AMGR in this fiscal year
            metrics = self._calculate_amgr_annual_metrics(group, amgr, fiscal_year, jimu_df)
            if metrics:
                annual_metrics.append(metrics)
        
        annual_metrics_df = pd.DataFrame(annual_metrics)
        
        # Calculate next year's disciplinary actions (target variable)
        annual_metrics_df = self._add_next_year_shobun_flag(annual_metrics_df, df)
        
        print(f"Annual metrics calculated: {len(annual_metrics_df)} records")
        self.annual_metrics_df = annual_metrics_df
        
        return annual_metrics_df
    
    def _calculate_amgr_annual_metrics(self, group: pd.DataFrame, amgr: str, 
                                     fiscal_year: int, jimu_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate annual metrics for a specific AMGR."""
        
        # Filter group to only LP records (RANK=10)
        lp_records = group[group['RANK'] == 10]
        
        if len(lp_records) == 0:
            return None
        
        total_lp_months = len(lp_records)
        
        # 1. MTG出席率 (Meeting Attendance Rate)
        mtg_attendance_count = lp_records['MTG_ATTENDANCE'].notna().sum()
        mtg_attendance_rate = mtg_attendance_count / total_lp_months if total_lp_months > 0 else 0
        
        # 2. 苦情率 (Complaint Rate)  
        complaint_count = lp_records['苦情'].notna().sum()
        complaint_rate = complaint_count / total_lp_months if total_lp_months > 0 else 0
        
        # 3. コンプライアンス違反疑義率 (Compliance Violation Rate)
        compliance_violation_count = (lp_records['コンプライアンス'] == 1).sum()
        compliance_rate = compliance_violation_count / total_lp_months if total_lp_months > 0 else 0
        
        # 4. 事務ミス率 (Administrative Error Rate)
        jimu_miss_rate = 0
        if not jimu_df.empty:
            # Check if any LP under this AMGR has 事務ミス records in this fiscal year
            lp_list = lp_records['LP'].unique()
            
            # Identify LP column in jimu_df
            lp_col_jimu = None
            for col in jimu_df.columns:
                if any(keyword in str(col).upper() for keyword in ['LP', 'SYAIN', 'CODE']):
                    lp_col_jimu = col
                    break
            
            if lp_col_jimu and 'Fiscal_Year' in jimu_df.columns:
                jimu_records = jimu_df[
                    (jimu_df[lp_col_jimu].isin(lp_list)) & 
                    (jimu_df['Fiscal_Year'] == fiscal_year)
                ]
                jimu_miss_count = len(jimu_records)
                jimu_miss_rate = jimu_miss_count / total_lp_months if total_lp_months > 0 else 0
        
        return {
            'AMGR': amgr,
            'Fiscal_Year': fiscal_year,
            'MTG_Attendance_Rate': mtg_attendance_rate,
            '苦情_Rate': complaint_rate,
            'コンプライアンス_Rate': compliance_rate,
            '事務ミス_Rate': jimu_miss_rate,
            'Total_LP_Months': total_lp_months,
            'Unique_LPs': lp_records['LP'].nunique()
        }
    
    def _add_next_year_shobun_flag(self, annual_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
        """Add next year's disciplinary action flag as target variable."""
        
        annual_df['SHOBUN_Flag'] = 0
        
        for idx, row in annual_df.iterrows():
            amgr = row['AMGR']
            current_year = row['Fiscal_Year']
            next_year = current_year + 1
            
            # Check if this AMGR has disciplinary actions in the next fiscal year
            next_year_data = main_df[
                (main_df['AMGR'] == amgr) & 
                (main_df['Fiscal_Year'] == next_year) &
                (main_df['SHOBUN'].notna())
            ]
            
            if len(next_year_data) > 0:
                annual_df.loc[idx, 'SHOBUN_Flag'] = 1
        
        return annual_df

    def validate_hypothesis_6_multidimensional_risk(self, df: pd.DataFrame,
                                                   jimu_miss_path: str = None) -> Dict[str, Any]:
        """
        Validate Hypothesis 6: Multi-dimensional Risk Factor Analysis

        Hypothesis: if "(变数1, 连续值)" & (变数2, 连续值) & (变数3, 连续值) & (变数4, 连续值), then (变数5)=1
        - 变数1: MTG出席率 (Meeting attendance rate)
        - 变数2: 苦情率 (Complaint rate)
        - 变数3: コンプライアンス違反疑義 (Compliance violation suspicion)
        - 变数4: 事務ミス (Administrative errors)
        - 变数5: 懲戒処分が出る (Disciplinary action occurs)

        Args:
            df: DataFrame containing the complete merged dataset
            jimu_miss_path: Path to administrative error data file

        Returns:
            Dictionary containing validation results
        """
        print("Validating Hypothesis 6: Multi-dimensional Risk Factor Analysis")

        # Register the hypothesis
        self.register_hypothesis(
            "H6_multidimensional_risk",
            "Multi-dimensional Risk: 4 continuous risk factors predict disciplinary actions",
            'Null: Risk factors do not predict disciplinary actions',
            'Alternative: Combination of 4 risk factors predicts disciplinary actions'
        )

        # Calculate annual metrics with all 4 risk factors
        annual_metrics_df = self.calculate_annual_metrics(df, jimu_miss_path)

        if annual_metrics_df.empty:
            return {'error': 'No annual metrics calculated'}

        # Train and evaluate machine learning model
        model_results = self._train_multidimensional_model(annual_metrics_df)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(annual_metrics_df)

        # Analyze feature correlations
        correlation_analysis = self._analyze_feature_correlations(annual_metrics_df)

        # Analyze feature importance
        feature_importance = self._analyze_feature_importance()

        # SHAP analysis if available
        shap_analysis = self._analyze_shap_values(annual_metrics_df)

        # Statistical tests
        statistical_tests = self._perform_statistical_tests(annual_metrics_df)

        # Compile validation results
        validation_results = {
            'hypothesis_id': 'H6_multidimensional_risk',
            'data_shape': annual_metrics_df.shape,
            'performance_metrics': performance_metrics,
            'model_results': model_results,
            'correlation_analysis': correlation_analysis,
            'feature_importance': feature_importance,
            'shap_analysis': shap_analysis,
            'statistical_tests': statistical_tests,
            'multidimensional_analysis': {
                'total_amgr_years': len(annual_metrics_df),
                'amgrs_with_disciplinary_actions': (annual_metrics_df['SHOBUN_Flag'] == 1).sum(),
                'disciplinary_action_rate': annual_metrics_df['SHOBUN_Flag'].mean(),
                'prediction_accuracy': model_results.get('accuracy', 0),
                'f1_score': model_results.get('f1_score', 0),
                'feature_count': 4
            },
            'conclusion': self._interpret_h6_results(annual_metrics_df, model_results, feature_importance),
            'validated_at': pd.Timestamp.now()
        }

        # Save results
        self.validation_results['H6_multidimensional_risk'] = validation_results

        print("Hypothesis 6 validation completed")
        return validation_results

    def _train_multidimensional_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train machine learning model for multidimensional risk prediction."""
        print("Training multidimensional risk prediction model...")

        if df.empty or len(df) < 10:
            return {'error': 'Insufficient data for model training'}

        # Prepare features and target
        feature_columns = ['MTG_Attendance_Rate', '苦情_Rate', 'コンプライアンス_Rate', '事務ミス_Rate']

        # Check if all feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            return {'error': f'Missing feature columns: {missing_features}'}

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
                max_depth=3,
                learning_rate=0.1,
                n_estimators=100,
                objective='binary:logistic',
                random_state=42
            )
        else:
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(
                max_depth=3,
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
            'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
            'feature_columns': feature_columns
        }

    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance metrics for the analysis results."""
        if df.empty:
            return {'error': 'No data available'}

        metrics = {}

        feature_columns = ['MTG_Attendance_Rate', '苦情_Rate', 'コンプライアンス_Rate', '事務ミス_Rate']

        for col in feature_columns + ['SHOBUN_Flag']:
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

    def _analyze_feature_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between features."""
        if df.empty:
            return {'error': 'No data available'}

        feature_columns = ['MTG_Attendance_Rate', '苦情_Rate', 'コンプライアンス_Rate', '事務ミス_Rate', 'SHOBUN_Flag']
        available_columns = [col for col in feature_columns if col in df.columns]

        if len(available_columns) < 2:
            return {'error': 'Insufficient columns for correlation analysis'}

        correlation_matrix = df[available_columns].corr()

        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': self._find_high_correlations(correlation_matrix),
            'target_correlations': correlation_matrix['SHOBUN_Flag'].to_dict() if 'SHOBUN_Flag' in correlation_matrix.columns else {}
        }

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find pairs of features with high correlation."""
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })

        return high_corr_pairs

    def _analyze_feature_importance(self) -> Dict[str, float]:
        """Analyze feature importance from the trained model."""
        if self.feature_importance is None:
            return {}

        return self.feature_importance

    def _analyze_shap_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SHAP values for feature interpretability."""
        if not SHAP_AVAILABLE or self.model is None:
            return {'error': 'SHAP not available or model not trained'}

        try:
            feature_columns = ['MTG_Attendance_Rate', '苦情_Rate', 'コンプライアンス_Rate', '事務ミス_Rate']
            available_columns = [col for col in feature_columns if col in df.columns]

            if len(available_columns) < 4:
                return {'error': 'Not all feature columns available for SHAP analysis'}

            X = df[available_columns]

            # Create SHAP explainer
            if XGBOOST_AVAILABLE and isinstance(self.model, XGBClassifier):
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.Explainer(self.model, X)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            self.shap_values = shap_values

            # Calculate mean absolute SHAP values for feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, take positive class

            mean_shap_values = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(available_columns, mean_shap_values))

            return {
                'shap_importance': shap_importance,
                'shap_values_shape': shap_values.shape,
                'feature_columns': available_columns
            }

        except Exception as e:
            return {'error': f'SHAP analysis failed: {str(e)}'}

    def _perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical tests on the features."""
        if df.empty:
            return {'error': 'No data available'}

        results = {}

        feature_columns = ['MTG_Attendance_Rate', '苦情_Rate', 'コンプライアンス_Rate', '事務ミス_Rate']

        # Test each feature against target
        for feature in feature_columns:
            if feature in df.columns and 'SHOBUN_Flag' in df.columns:
                # Split data by target
                group_0 = df[df['SHOBUN_Flag'] == 0][feature]
                group_1 = df[df['SHOBUN_Flag'] == 1][feature]

                if len(group_0) > 0 and len(group_1) > 0:
                    # Perform t-test
                    from scipy.stats import ttest_ind
                    t_stat, p_value = ttest_ind(group_0, group_1)

                    results[feature] = {
                        'group_0_mean': group_0.mean(),
                        'group_1_mean': group_1.mean(),
                        'group_0_std': group_0.std(),
                        'group_1_std': group_1.std(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }

        return results

    def _interpret_h6_results(self, df: pd.DataFrame, model_results: Dict[str, Any],
                             feature_importance: Dict[str, float]) -> str:
        """Interpret the results of Hypothesis 6 validation."""
        if df.empty:
            return "Error: No data available for interpretation"

        # Calculate key metrics
        total_records = len(df)
        with_disciplinary = (df['SHOBUN_Flag'] == 1).sum()
        disciplinary_rate = with_disciplinary / total_records if total_records > 0 else 0

        accuracy = model_results.get('accuracy', 0)
        f1 = model_results.get('f1_score', 0)
        model_type = model_results.get('model_type', 'Unknown')

        # Find most important feature
        most_important_feature = 'Unknown'
        if feature_importance:
            most_important_feature = max(feature_importance.keys(), key=lambda k: feature_importance[k])

        conclusion = f"Hypothesis 6 Results: {total_records} AMGR-year records analyzed, "
        conclusion += f"{with_disciplinary} with disciplinary actions ({disciplinary_rate:.3f} rate). "
        conclusion += f"{model_type} model accuracy: {accuracy:.3f}, F1-score: {f1:.3f}. "
        conclusion += f"Most important risk factor: {most_important_feature}. "
        conclusion += f"Multi-dimensional risk model {'successful' if accuracy > 0.6 else 'needs improvement'}."

        return conclusion
