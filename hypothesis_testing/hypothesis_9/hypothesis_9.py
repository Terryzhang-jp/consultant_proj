"""
Hypothesis 9 Validator: Low Salary and Turnover Analysis

This module validates Hypothesis 9:
"If the number of LPs with low salary (SASHIHIKI < 200,000) for 6 consecutive months 
and who left within 2 years is >= 1, then disciplinary actions occur in the company."

Hypothesis: if "(变数3) >= 1" then (变数4) = 1
- 变数1(t): "低報酬(SASHIHIKI20万円未満)が6か月継続" (Low salary < 200,000 yen for 6 consecutive months)
- 变数2(t): "LPが2年以内に退職" (LP leaves within 2 years)
- 变数3(t): 变数1(t) & 变数2(t) LP 総人数 (Total number of LPs meeting both conditions)
- 变数4(t): その会社での懲戒処分の有無 (Presence of disciplinary actions in the company)
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


class Hypothesis9Validator(BaseHypothesisValidator):
    """
    Hypothesis 9 验证器：低薪酬和离职分析
    
    验证假设：if "(变数3) >= 1" then (变数4) = 1
    - 变数1(t): 低報酬(SASHIHIKI20万円未満)が6か月継続
    - 变数2(t): LPが2年以内に退職
    - 变数3(t): 变数1(t) & 变数2(t) LP 総人数
    - 变数4(t): その会社での懲戒処分の有無
    """
    
    def __init__(self):
        super().__init__()
        self.xgb_analyzer = None

    def verify_hypothesis_adjusted(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        验证假设的核心方法，改进了离职日期确定和总LP计算

        Args:
            monthly_df: 包含LP月度数据的DataFrame

        Returns:
            amgr_summary: 汇总各营业所的总LP数、符合条件的LP数和纪律处分数的DataFrame
        """
        print("Starting Hypothesis 9 verification...")

        # Step 1: 过滤所有AMGR (rank 20) 代表営業所
        amgr_df = monthly_df[monthly_df['RANK_x'] == 20].drop_duplicates(subset='AMGR')
        print(f"Found {len(amgr_df)} unique AMGRs (営業所)")

        # Step 2: 过滤各営業所的LP (rank 10)
        lp_df = monthly_df[monthly_df['RANK_x'] == 10].copy()
        print(f"Found {len(lp_df)} LP records")

        # Step 3: 识别SASHIHIKI < 200,000连续6个月的LP
        def filter_low_sashihiki(group):
            """识别低薪酬连续6个月的LP"""
            group = group.sort_values('Date') if 'Date' in group.columns else group.sort_values(['S_YR', 'S_MO'])
            group['low_sashihiki'] = group['SASHIHIKI'] < 200000
            group['low_sashihiki_streak'] = group['low_sashihiki'].rolling(
                window=6, min_periods=6).sum() == 6
            return group

        # 为每个LP应用过滤函数
        lp_df = lp_df.reset_index(drop=True).groupby(
            'LP', group_keys=False).apply(filter_low_sashihiki)

        print(f"Processed low SASHIHIKI filtering for {lp_df['LP'].nunique()} unique LPs")

        # Step 4: 过滤在低SASHIHIKI连续期后2年内离职的LP
        def filter_left_within_two_years(group):
            """识别2年内离职的LP"""
            if group['low_sashihiki_streak'].any():
                streak_start_date = group.loc[group['low_sashihiki_streak'], 'Date'].min() if 'Date' in group.columns else None

                # 基于STATUS == 'T'确定离职日期
                if 'STATUS' in group.columns:
                    leave_date = group.loc[group['STATUS'] == 'T', 'Date'].max() if (group['STATUS'] == 'T').any() else None
                else:
                    # 如果没有STATUS列，使用最后记录日期作为离职日期
                    leave_date = group['Date'].max() if 'Date' in group.columns else None

                if leave_date is not None and streak_start_date is not None:
                    group['left_within_2_years'] = (leave_date - streak_start_date).days <= 730  # 2年
                else:
                    group['left_within_2_years'] = False
            else:
                group['left_within_2_years'] = False
            return group

        lp_df = lp_df.reset_index(drop=True).groupby(
            'LP', group_keys=False).apply(filter_left_within_two_years)

        # Step 5: 统计各営業所符合条件的LP数量
        def count_conditions(group):
            """统计各种条件的LP数量"""
            # 営業所内总LP数（考虑每个AMGR内的唯一LP）
            total_lps = group['LP'].nunique()

            # 统计低SASHIHIKI且2年内离职的LP数
            low_sashihiki_and_left = group[
                (group['low_sashihiki_streak']) &
                (group['left_within_2_years'])
            ]['LP'].nunique()

            # 统计纪律处分数量
            shobun_count = group['SHOBUN'].notna().sum() if 'SHOBUN' in group.columns else 0

            return pd.Series({
                'total_lps': total_lps,
                'low_sashihiki_and_left': low_sashihiki_and_left,
                'shobun_count': shobun_count
            })

        # 按営業所(AMGR)分组并计算统计数据
        amgr_summary = lp_df.groupby('AMGR').apply(count_conditions).reset_index()

        print(f"Generated summary for {len(amgr_summary)} AMGRs")

        return amgr_summary

    def validate_hypothesis_9_low_salary_turnover(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        验证 Hypothesis 9: 低薪酬和离职分析

        Hypothesis: if "(变数3) >= 1" then (变数4) = 1
        - 变数1(t): 低報酬(SASHIHIKI20万円未満)が6か月継続
        - 变数2(t): LPが2年以内に退職
        - 变数3(t): 变数1(t) & 变数2(t) LP 総人数
        - 变数4(t): その会社での懲戒処分の有無

        Args:
            df: 包含完整数据的DataFrame

        Returns:
            验证结果字典
        """
        print("Validating Hypothesis 9: Low Salary and Turnover Analysis")

        # 注册假设
        self.register_hypothesis(
            "H9_low_salary_turnover",
            "Low Salary and Turnover: Low salary LPs leaving predicts disciplinary actions",
            'Null: Low salary turnover does not predict disciplinary actions',
            'Alternative: LPs with low salary leaving within 2 years predicts disciplinary actions'
        )

        # 执行核心验证逻辑
        amgr_summary = self.verify_hypothesis_adjusted(df)

        if amgr_summary.empty:
            return {'error': 'No analysis data prepared'}

        # 数据清理：移除异常值
        # 移除shobun_count为125的记录（如原代码所示）
        amgr_summary = amgr_summary[amgr_summary['shobun_count'] != 125]

        # 计算纪律处分率
        amgr_summary['sho_bun_rate'] = amgr_summary['shobun_count'] / amgr_summary['total_lps']

        # 创建标志变量
        amgr_summary['sho_bun_rate_flag'] = (amgr_summary['sho_bun_rate'] > 0).astype(int)
        amgr_summary['low_sashihiki_and_left_flag'] = (amgr_summary['low_sashihiki_and_left'] > 0).astype(int)

        # 验证模式1: XGBoost机器学习验证
        xgb_results = self._validate_with_xgboost(amgr_summary)

        # 验证模式2: 简单验证模式（混淆矩阵）
        simple_results = self._validate_with_simple_method(amgr_summary)

        # 统计分析
        statistical_analysis = self._perform_statistical_analysis(amgr_summary)

        # 描述性统计
        descriptive_stats = self._calculate_descriptive_statistics(amgr_summary)

        # 编译验证结果
        validation_results = {
            'hypothesis_id': 'H9_low_salary_turnover',
            'data_shape': amgr_summary.shape,
            'descriptive_statistics': descriptive_stats,
            'xgboost_validation': xgb_results,
            'simple_validation': simple_results,
            'statistical_analysis': statistical_analysis,
            'low_salary_turnover_analysis': {
                'total_amgr_records': len(amgr_summary),
                'amgr_with_low_salary_turnover': (amgr_summary['low_sashihiki_and_left'] >= 1).sum(),
                'amgr_with_disciplinary_actions': (amgr_summary['sho_bun_rate_flag'] == 1).sum(),
                'hypothesis_support_rate': self._calculate_hypothesis_support_rate(amgr_summary)
            },
            'conclusion': self._interpret_h9_results(amgr_summary, xgb_results, simple_results),
            'validated_at': pd.Timestamp.now()
        }

        # 保存结果
        self.validation_results['H9_low_salary_turnover'] = validation_results

        print("Hypothesis 9 validation completed")
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
            X = df[['low_sashihiki_and_left']]
            y = df['sho_bun_rate_flag']

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
            # 创建混淆矩阵：low_sashihiki_and_left_flag vs sho_bun_rate_flag
            cm = confusion_matrix(
                df['sho_bun_rate_flag'],
                df['low_sashihiki_and_left_flag']
            )

            # 计算F1分数
            f1 = f1_score(
                df['sho_bun_rate_flag'],
                df['low_sashihiki_and_left_flag']
            )

            # 计算准确率
            accuracy = accuracy_score(
                df['sho_bun_rate_flag'],
                df['low_sashihiki_and_left_flag']
            )

            # 计算分类报告
            class_report = classification_report(
                df['sho_bun_rate_flag'],
                df['low_sashihiki_and_left_flag'],
                output_dict=True
            )

            return {
                'confusion_matrix': cm.tolist(),
                'f1_score': f1,
                'accuracy': accuracy,
                'classification_report': class_report,
                'method': 'Simple binary classification based on low_sashihiki_and_left >= 1'
            }

        except Exception as e:
            return {'error': f'Simple validation failed: {str(e)}'}

    def _perform_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """执行统计分析。"""
        results = {}

        try:
            # 卡方检验：低薪离职标志 vs 纪律处分标志
            if SCIPY_AVAILABLE:
                contingency_table = pd.crosstab(
                    df['low_sashihiki_and_left_flag'],
                    df['sho_bun_rate_flag']
                )
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                results['chi_square_test'] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'significant': p_value < 0.05,
                    'contingency_table': contingency_table.to_dict()
                }

            # t检验：比较有无纪律处分组的低薪离职数量
            if SCIPY_AVAILABLE:
                group_0 = df[df['sho_bun_rate_flag'] == 0]['low_sashihiki_and_left']
                group_1 = df[df['sho_bun_rate_flag'] == 1]['low_sashihiki_and_left']

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
            correlation = df['low_sashihiki_and_left'].corr(df['sho_bun_rate_flag'])
            results['correlation'] = {
                'low_salary_turnover_vs_disciplinary': correlation
            }

        except Exception as e:
            results['error'] = f'Statistical analysis failed: {str(e)}'

        return results

    def _calculate_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算描述性统计。"""
        stats = {}

        # 低薪离职统计
        stats['low_salary_turnover'] = {
            'mean': df['low_sashihiki_and_left'].mean(),
            'median': df['low_sashihiki_and_left'].median(),
            'std': df['low_sashihiki_and_left'].std(),
            'min': df['low_sashihiki_and_left'].min(),
            'max': df['low_sashihiki_and_left'].max(),
            'count_with_turnover': (df['low_sashihiki_and_left'] > 0).sum(),
            'rate_with_turnover': (df['low_sashihiki_and_left'] > 0).mean()
        }

        # 纪律处分统计
        stats['disciplinary_actions'] = {
            'total_amgr': len(df),
            'with_disciplinary_actions': (df['sho_bun_rate_flag'] == 1).sum(),
            'disciplinary_action_rate': df['sho_bun_rate_flag'].mean(),
            'average_shobun_rate': df['sho_bun_rate'].mean(),
            'max_shobun_rate': df['sho_bun_rate'].max()
        }

        # 总LP统计
        stats['total_lps'] = {
            'mean_lps_per_amgr': df['total_lps'].mean(),
            'median_lps_per_amgr': df['total_lps'].median(),
            'total_lps_all_amgr': df['total_lps'].sum()
        }

        return stats

    def _calculate_hypothesis_support_rate(self, df: pd.DataFrame) -> float:
        """
        计算假设支持率。

        假设: if "(变数3) >= 1" then (变数4) = 1
        支持率 = (低薪离职>=1 且 有纪律处分) / (低薪离职>=1)
        """
        amgr_with_turnover = df[df['low_sashihiki_and_left'] >= 1]

        if len(amgr_with_turnover) == 0:
            return 0.0

        support_count = (amgr_with_turnover['sho_bun_rate_flag'] == 1).sum()
        support_rate = support_count / len(amgr_with_turnover)

        return support_rate

    def _interpret_h9_results(self, df: pd.DataFrame, xgb_results: Dict[str, Any],
                             simple_results: Dict[str, Any]) -> str:
        """解释Hypothesis 9的验证结果。"""
        if df.empty:
            return "Error: No data available for interpretation"

        # 基本统计
        total_amgr = len(df)
        amgr_with_turnover = (df['low_sashihiki_and_left'] >= 1).sum()
        turnover_rate = amgr_with_turnover / total_amgr if total_amgr > 0 else 0
        disciplinary_count = (df['sho_bun_rate_flag'] == 1).sum()
        disciplinary_rate = disciplinary_count / total_amgr if total_amgr > 0 else 0

        # 假设支持率
        support_rate = self._calculate_hypothesis_support_rate(df)

        # XGBoost结果
        xgb_accuracy = xgb_results.get('accuracy', 0) if 'error' not in xgb_results else 0
        xgb_f1 = xgb_results.get('f1_score', 0) if 'error' not in xgb_results else 0

        # 简单验证结果
        simple_accuracy = simple_results.get('accuracy', 0) if 'error' not in simple_results else 0
        simple_f1 = simple_results.get('f1_score', 0) if 'error' not in simple_results else 0

        # 平均统计
        avg_turnover = df['low_sashihiki_and_left'].mean()
        avg_shobun_rate = df['sho_bun_rate'].mean()

        conclusion = f"Hypothesis 9 Results: {total_amgr} AMGR records analyzed. "
        conclusion += f"Average low-salary turnover per AMGR: {avg_turnover:.2f}. "
        conclusion += f"AMGRs with low-salary turnover (>=1): {amgr_with_turnover} ({turnover_rate:.3f} rate). "
        conclusion += f"AMGRs with disciplinary actions: {disciplinary_count} ({disciplinary_rate:.3f} rate). "
        conclusion += f"Average disciplinary rate: {avg_shobun_rate:.3f}. "
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
