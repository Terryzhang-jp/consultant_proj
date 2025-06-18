"""
Hypothesis 10 Validator: AMGR Turnover and Disciplinary Actions Analysis

This module validates Hypothesis 10:
"If AMGR (営業所長) changes 3 or more times within 1 year in an office, 
then disciplinary actions occur in the following year."

Hypothesis: if "(变数1) = 1 & 变数2(t)" then (变数3) = 1
- 变数1(t): "営業所長(AMGR)が1年以内" (AMGR within 1 year)
- 变数2(t): "3回以上変わる" (Changes 3 or more times)
- 变数3(t): "オフィス(支社)の翌年の懲戒処分の有無" (Disciplinary actions in following year)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta

# Import base validator
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
try:
    from base_validator import BaseHypothesisValidator
except ImportError:
    # Fallback to a simple base class if base_validator is not available
    class BaseHypothesisValidator:
        def __init__(self):
            self.validation_results = {}
            self.registered_hypotheses = {}
        
        def register_hypothesis(self, hypothesis_id: str, description: str, 
                              null_hypothesis: str, alternative_hypothesis: str):
            """Register a hypothesis for validation."""
            self.registered_hypotheses[hypothesis_id] = {
                'description': description,
                'null_hypothesis': null_hypothesis,
                'alternative_hypothesis': alternative_hypothesis,
                'registered_at': pd.Timestamp.now()
            }

# Machine learning imports
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some features will be limited.")

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Using sklearn DecisionTreeClassifier instead.")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Class balancing disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Feature importance analysis limited.")

# Statistical imports
try:
    from scipy.stats import ttest_ind, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Statistical tests limited.")


class XGBoostAnalyzer:
    """XGBoost分析器，用于机器学习验证模式。"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.feature_importance = None
        self.shap_values = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    balance_method: str = 'smote', test_size: float = 0.2) -> Tuple:
        """准备训练和测试数据。"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for data preparation")
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # 类平衡处理
        if balance_method == 'smote' and SMOTE_AVAILABLE and len(y_train.unique()) > 1:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict = None) -> None:
        """训练XGBoost模型。"""
        if params is None:
            params = {
                'max_depth': 2,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': self.random_state
            }
        
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(**params)
        else:
            # 使用DecisionTree作为备选
            self.model = DecisionTreeClassifier(
                max_depth=params.get('max_depth', 2),
                random_state=self.random_state,
                class_weight='balanced'
            )
        
        self.model.fit(X_train, y_train)
        
        # 保存特征重要度
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X_train.columns, self.model.feature_importances_))
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """评估模型性能。"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'feature_importance': self.feature_importance
        }
        
        return results
    
    def analyze_shap(self, X: pd.DataFrame, plot_type: str = 'beeswarm') -> Dict[str, Any]:
        """SHAP分析。"""
        if not SHAP_AVAILABLE or self.model is None:
            return {'error': 'SHAP not available or model not trained'}
        
        try:
            # 创建SHAP解释器
            if XGBOOST_AVAILABLE and isinstance(self.model, XGBClassifier):
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.Explainer(self.model, X)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X)
            self.shap_values = shap_values
            
            # 计算平均绝对SHAP值
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 二分类取正类
            
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(X.columns, mean_shap_values))
            
            return {
                'shap_importance': shap_importance,
                'shap_values_shape': shap_values.shape,
                'feature_columns': list(X.columns)
            }
            
        except Exception as e:
            return {'error': f'SHAP analysis failed: {str(e)}'}


class Hypothesis10Validator(BaseHypothesisValidator):
    """
    Hypothesis 10 验证器：AMGR变更和纪律处分分析
    
    验证假设：if "(变数1) = 1 & 变数2(t)" then (变数3) = 1
    - 变数1(t): 営業所長(AMGR)が1年以内
    - 变数2(t): 3回以上変わる
    - 变数3(t): オフィス(支社)の翌年の懲戒処分の有無
    """
    
    def __init__(self):
        super().__init__()
        self.xgb_analyzer = None

    def analyze_office_amgr_changes_corrected(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        修正后的AMGR变更分析方法

        正确计算AMGR总变更次数（新增 + 离开），而不是只计算新增

        Args:
            dataframe: 包含OFFICE, S_YR, AMGR, LP, SHOBUN列的DataFrame

        Returns:
            annual_office_df: 包含年度办公室AMGR变更和纪律处分数据的DataFrame
        """
        print("Starting corrected AMGR change analysis...")

        # 去重：按LP, S_YR, S_MO去重
        df_cleaned = dataframe.drop_duplicates(subset=['LP', 'S_YR', 'S_MO'])
        print(f"Cleaned data shape: {df_cleaned.shape}")

        # 按办公室和年份分组
        office_groups = df_cleaned.groupby(['OFFICE', 'S_YR'])

        result = []
        office_amgr_history = {}  # 存储每个办公室的AMGR历史

        for (office, year), group in office_groups:
            # 获取当年的所有AMGR
            current_amgrs = set(group['AMGR'].unique())

            # 获取前一年的AMGR（如果存在）
            if office in office_amgr_history:
                prev_amgrs = office_amgr_history[office]
            else:
                prev_amgrs = set()  # 第一年没有前一年数据

            # 修正后的变更计算逻辑
            new_amgrs = current_amgrs - prev_amgrs      # 新增的AMGR
            left_amgrs = prev_amgrs - current_amgrs     # 离开的AMGR
            total_changes = len(new_amgrs) + len(left_amgrs)  # 总变更次数

            # 判断是否为高变更年（>=3次变更）
            is_high_change = total_changes >= 3

            # 计算办公室当年总LP数
            total_lps = group['LP'].nunique()

            # 计算当年SHOBUN数量
            shobun_count = group['SHOBUN'].notna().sum()

            # 记录结果
            result.append({
                'OFFICE': office,
                'YEAR': year,
                'CURRENT_AMGRS': list(current_amgrs),
                'PREV_AMGRS': list(prev_amgrs),
                'NEW_AMGRS': list(new_amgrs),
                'LEFT_AMGRS': list(left_amgrs),
                'TOTAL_CHANGES': total_changes,
                'AMGR_CHANGE_FLAG': is_high_change,  # 变数1&2: AMGR在1年内变更>=3次
                'TOTAL_LPS': total_lps,
                'SHOBUN_COUNT': shobun_count
            })

            # 更新办公室AMGR历史
            office_amgr_history[office] = current_amgrs

        # 转换为DataFrame
        annual_office_df = pd.DataFrame(result)

        # 计算次年纪律处分（变数3）
        annual_office_df['SHOBUN_COUNT_NEXT_YEAR'] = annual_office_df.groupby('OFFICE')['SHOBUN_COUNT'].shift(-1)
        annual_office_df['SHOBUN_FLAG_NEXT_YEAR'] = (annual_office_df['SHOBUN_COUNT_NEXT_YEAR'] > 0).astype(int)

        # 移除没有次年数据的记录
        annual_office_df = annual_office_df.dropna(subset=['SHOBUN_COUNT_NEXT_YEAR'])

        print(f"Generated {len(annual_office_df)} office-year records for analysis")

        return annual_office_df

    def validate_hypothesis_10_amgr_turnover(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证 Hypothesis 10: AMGR变更和纪律处分分析

        Hypothesis: if "(变数1) = 1 & 变数2(t)" then (变数3) = 1
        - 变数1(t): 営業所長(AMGR)が1年以内
        - 变数2(t): 3回以上変わる
        - 变数3(t): オフィス(支社)の翌年の懲戒処分の有無

        Args:
            df: 包含完整数据的DataFrame

        Returns:
            验证结果字典
        """
        print("Validating Hypothesis 10: AMGR Turnover and Disciplinary Actions")

        # 注册假设
        self.register_hypothesis(
            "H10_amgr_turnover",
            "AMGR Turnover: High AMGR turnover predicts disciplinary actions in following year",
            'Null: AMGR turnover does not predict disciplinary actions in following year',
            'Alternative: High AMGR turnover (>=3 changes) predicts disciplinary actions in following year'
        )

        # 执行修正后的AMGR变更分析
        annual_office_df = self.analyze_office_amgr_changes_corrected(df)

        if annual_office_df.empty:
            return {'error': 'No analysis data prepared'}

        # 验证模式1: XGBoost机器学习验证
        xgb_results = self._validate_with_xgboost(annual_office_df)

        # 验证模式2: 简单验证模式（混淆矩阵）
        simple_results = self._validate_with_simple_method(annual_office_df)

        # 统计分析
        statistical_analysis = self._perform_statistical_analysis(annual_office_df)

        # 描述性统计
        descriptive_stats = self._calculate_descriptive_statistics(annual_office_df)

        # 编译验证结果
        validation_results = {
            'hypothesis_id': 'H10_amgr_turnover',
            'data_shape': annual_office_df.shape,
            'descriptive_statistics': descriptive_stats,
            'xgboost_validation': xgb_results,
            'simple_validation': simple_results,
            'statistical_analysis': statistical_analysis,
            'amgr_turnover_analysis': {
                'total_office_year_records': len(annual_office_df),
                'offices_with_high_amgr_turnover': (annual_office_df['AMGR_CHANGE_FLAG'] == 1).sum(),
                'offices_with_next_year_disciplinary': (annual_office_df['SHOBUN_FLAG_NEXT_YEAR'] == 1).sum(),
                'hypothesis_support_rate': self._calculate_hypothesis_support_rate(annual_office_df)
            },
            'conclusion': self._interpret_h10_results(annual_office_df, xgb_results, simple_results),
            'validated_at': pd.Timestamp.now()
        }

        # 保存结果
        self.validation_results['H10_amgr_turnover'] = validation_results

        print("Hypothesis 10 validation completed")
        return validation_results

    def _validate_with_xgboost(self, df: pd.DataFrame) -> Dict[str, Any]:
        """使用XGBoost进行机器学习验证。"""
        print("Running XGBoost validation...")

        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available for XGBoost validation'}

        try:
            # 初始化XGBoost分析器
            self.xgb_analyzer = XGBoostAnalyzer(random_state=42)

            # 准备数据
            X = df[['AMGR_CHANGE_FLAG']]  # 特征：AMGR高变更标志
            y = df['SHOBUN_FLAG_NEXT_YEAR']  # 目标：次年纪律处分标志

            if len(y.unique()) < 2:
                return {'error': 'Target variable has only one class'}

            # 数据分割和平衡
            X_train, X_test, y_train, y_test = self.xgb_analyzer.prepare_data(
                X, y, balance_method='smote'
            )

            # 训练模型
            params = {
                'max_depth': 2,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'binary:logistic',
                'random_state': 42
            }
            self.xgb_analyzer.train(X_train, y_train, params=params)

            # 评估模型
            results = self.xgb_analyzer.evaluate(X_test, y_test)

            # SHAP分析
            shap_results = self.xgb_analyzer.analyze_shap(X)
            results['shap_analysis'] = shap_results

            return results

        except Exception as e:
            return {'error': f'XGBoost validation failed: {str(e)}'}

    def _validate_with_simple_method(self, df: pd.DataFrame) -> Dict[str, Any]:
        """使用简单方法进行验证（混淆矩阵）。"""
        print("Running simple validation method...")

        try:
            # 创建混淆矩阵：AMGR_CHANGE_FLAG vs SHOBUN_FLAG_NEXT_YEAR
            cm = confusion_matrix(
                df['SHOBUN_FLAG_NEXT_YEAR'],
                df['AMGR_CHANGE_FLAG']
            )

            # 计算F1分数
            f1 = f1_score(
                df['SHOBUN_FLAG_NEXT_YEAR'],
                df['AMGR_CHANGE_FLAG']
            )

            # 计算准确率
            accuracy = accuracy_score(
                df['SHOBUN_FLAG_NEXT_YEAR'],
                df['AMGR_CHANGE_FLAG']
            )

            # 计算分类报告
            class_report = classification_report(
                df['SHOBUN_FLAG_NEXT_YEAR'],
                df['AMGR_CHANGE_FLAG'],
                output_dict=True
            )

            return {
                'confusion_matrix': cm.tolist(),
                'f1_score': f1,
                'accuracy': accuracy,
                'classification_report': class_report,
                'method': 'Simple binary classification based on AMGR_CHANGE_FLAG'
            }

        except Exception as e:
            return {'error': f'Simple validation failed: {str(e)}'}

    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """执行统计分析。"""
        results = {}

        try:
            # 卡方检验：AMGR变更标志 vs 次年纪律处分标志
            if SCIPY_AVAILABLE:
                contingency_table = pd.crosstab(
                    df['AMGR_CHANGE_FLAG'],
                    df['SHOBUN_FLAG_NEXT_YEAR']
                )
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                results['chi_square_test'] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < 0.05,
                    'contingency_table': contingency_table.to_dict()
                }

            # t检验：比较高变更组和低变更组的次年纪律处分数量
            if SCIPY_AVAILABLE:
                group_0 = df[df['AMGR_CHANGE_FLAG'] == 0]['SHOBUN_COUNT_NEXT_YEAR']
                group_1 = df[df['AMGR_CHANGE_FLAG'] == 1]['SHOBUN_COUNT_NEXT_YEAR']

                if len(group_0) > 0 and len(group_1) > 0:
                    t_stat, t_p_value = ttest_ind(group_0, group_1)

                    results['t_test'] = {
                        'group_0_mean': group_0.mean(),
                        'group_1_mean': group_1.mean(),
                        'group_0_std': group_0.std(),
                        'group_1_std': group_1.std(),
                        't_statistic': t_stat,
                        'p_value': t_p_value,
                        'significant': t_p_value < 0.05
                    }

            # 相关性分析
            correlation = df['AMGR_CHANGE_FLAG'].corr(df['SHOBUN_FLAG_NEXT_YEAR'])
            results['correlation'] = {
                'amgr_turnover_vs_next_year_disciplinary': correlation
            }

        except Exception as e:
            results['error'] = f'Statistical analysis failed: {str(e)}'

        return results

    def _calculate_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算描述性统计。"""
        stats = {}

        # AMGR变更统计
        stats['amgr_changes'] = {
            'total_office_years': len(df),
            'high_change_office_years': (df['AMGR_CHANGE_FLAG'] == 1).sum(),
            'high_change_rate': df['AMGR_CHANGE_FLAG'].mean(),
            'average_total_changes': df['TOTAL_CHANGES'].mean(),
            'max_total_changes': df['TOTAL_CHANGES'].max(),
            'min_total_changes': df['TOTAL_CHANGES'].min()
        }

        # 次年纪律处分统计
        stats['next_year_disciplinary'] = {
            'office_years_with_next_disciplinary': (df['SHOBUN_FLAG_NEXT_YEAR'] == 1).sum(),
            'next_year_disciplinary_rate': df['SHOBUN_FLAG_NEXT_YEAR'].mean(),
            'average_next_year_shobun': df['SHOBUN_COUNT_NEXT_YEAR'].mean(),
            'max_next_year_shobun': df['SHOBUN_COUNT_NEXT_YEAR'].max()
        }

        # 办公室LP统计
        stats['office_lps'] = {
            'average_lps_per_office_year': df['TOTAL_LPS'].mean(),
            'median_lps_per_office_year': df['TOTAL_LPS'].median(),
            'total_unique_offices': df['OFFICE'].nunique()
        }

        return stats

    def _calculate_hypothesis_support_rate(self, df: pd.DataFrame) -> float:
        """
        计算假设支持率。

        假设: if AMGR_CHANGE_FLAG = 1 then SHOBUN_FLAG_NEXT_YEAR = 1
        支持率 = (高AMGR变更 且 次年有纪律处分) / (高AMGR变更)
        """
        high_change_offices = df[df['AMGR_CHANGE_FLAG'] == 1]

        if len(high_change_offices) == 0:
            return 0.0

        support_count = (high_change_offices['SHOBUN_FLAG_NEXT_YEAR'] == 1).sum()
        support_rate = support_count / len(high_change_offices)

        return support_rate

    def _interpret_h10_results(self, df: pd.DataFrame, xgb_results: Dict[str, Any],
                             simple_results: Dict[str, Any]) -> str:
        """解释Hypothesis 10的验证结果。"""
        if df.empty:
            return "Error: No data available for interpretation"

        # 基本统计
        total_office_years = len(df)
        high_change_count = (df['AMGR_CHANGE_FLAG'] == 1).sum()
        high_change_rate = high_change_count / total_office_years if total_office_years > 0 else 0
        next_year_disciplinary_count = (df['SHOBUN_FLAG_NEXT_YEAR'] == 1).sum()
        next_year_disciplinary_rate = next_year_disciplinary_count / total_office_years if total_office_years > 0 else 0

        # 假设支持率
        support_rate = self._calculate_hypothesis_support_rate(df)

        # XGBoost结果
        xgb_accuracy = xgb_results.get('accuracy', 0) if 'error' not in xgb_results else 0
        xgb_f1 = xgb_results.get('f1_score', 0) if 'error' not in xgb_results else 0

        # 简单验证结果
        simple_accuracy = simple_results.get('accuracy', 0) if 'error' not in simple_results else 0
        simple_f1 = simple_results.get('f1_score', 0) if 'error' not in simple_results else 0

        # 平均统计
        avg_total_changes = df['TOTAL_CHANGES'].mean()
        avg_next_year_shobun = df['SHOBUN_COUNT_NEXT_YEAR'].mean()

        conclusion = f"Hypothesis 10 Results: {total_office_years} office-year records analyzed. "
        conclusion += f"Average AMGR changes per office-year: {avg_total_changes:.2f}. "
        conclusion += f"Office-years with high AMGR turnover (>=3 changes): {high_change_count} ({high_change_rate:.3f} rate). "
        conclusion += f"Office-years with next-year disciplinary actions: {next_year_disciplinary_count} ({next_year_disciplinary_rate:.3f} rate). "
        conclusion += f"Average next-year disciplinary count: {avg_next_year_shobun:.2f}. "
        conclusion += f"Hypothesis support rate: {support_rate:.3f}. "
        conclusion += f"XGBoost validation: accuracy={xgb_accuracy:.3f}, F1={xgb_f1:.3f}. "
        conclusion += f"Simple validation: accuracy={simple_accuracy:.3f}, F1={simple_f1:.3f}. "

        if support_rate > 0.6:
            conclusion += "Strong support for hypothesis."
        elif support_rate > 0.4:
            conclusion += "Moderate support for hypothesis."
        else:
            conclusion += "Weak support for hypothesis."

        return conclusion
