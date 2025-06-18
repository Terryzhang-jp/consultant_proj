# Hypothesis 9+10 Combined Validation

## 假设定义 / Hypothesis Definition

**组合假设 9+10**: "以下の2つの条件(#9 & #10)が揃うと、支社での翌年の懲戒処分の有無を受ける"

### 假设逻辑 / Hypothesis Logic
```
if "(变数1) = 1 & 变数2(t)" then 变数3(t)
```

### 变数定义 / Variable Definitions

- **变数1(t)** (Hypothesis 10): "営業所長が1年以内、3回以上変わる"
  - AMGR在1年内变更3次以上（包括新增和离开）
  - AMGR changes 3+ times within 1 year (including new arrivals and departures)

- **变数2(t)** (Hypothesis 9): "低報酬(SASHIHIKI20万円未満)が6か月継続、LPが2年以内に退職"
  - 低薪酬（SASHIHIKI < 20万日元）连续6个月，且LP在2年内离职
  - Low salary (SASHIHIKI < 200,000 yen) for 6+ consecutive months AND LP leaves within 2 years

- **变数3(t)**: "支社での懲戒処分の有無"
  - 支社次年是否有纪律处分
  - Whether disciplinary actions occur in the branch the following year

### 组合条件 / Combined Condition
```
Combined_Condition = H9_condition AND H10_condition
Prediction = Combined_Condition → Next_Year_Disciplinary_Actions
```

## 代码结构 / Code Structure

### 核心模块 / Core Modules

1. **`hypothesis_validator_9_10_combined.py`**
   - 主要验证器类 `Hypothesis9And10CombinedValidator`
   - 分别分析H9和H10条件
   - 组合条件验证和性能评估

2. **`ml_analysis.py`**
   - 机器学习分析模块 `MLAnalyzer`
   - XGBoost建模和交叉验证
   - SHAP特征重要性分析

3. **`test_hypothesis_9_10_combined.py`**
   - 测试脚本，包含模拟数据生成
   - 完整的验证流程测试

### 关键修正 / Key Corrections

#### ✅ 修正1: AMGR变更检测逻辑
```python
# 错误的逻辑 (原代码)
changed_amgrs = list(set(amgrs) - prev_amgrs)  # 只检测新增
is_high_change = len(changed_amgrs) >= 3

# 正确的逻辑 (修正后)
new_amgrs = current_amgrs - prev_amgrs        # 新增AMGR
departed_amgrs = prev_amgrs - current_amgrs   # 离开AMGR
total_changes = len(new_amgrs) + len(departed_amgrs)  # 总变更数
h10_condition_met = total_changes >= 3        # H10条件
```

#### ✅ 修正2: 组合逻辑
```python
# 错误的逻辑 (原代码)
combined_flag = (h9_condition | h10_condition)  # OR逻辑

# 正确的逻辑 (修正后)
combined_condition = (h9_condition & h10_condition)  # AND逻辑
```

#### ✅ 修正3: 变数定义明确化
- **H9条件**: 低薪酬连续6个月 AND 2年内离职的LP数量 > 0
- **H10条件**: AMGR总变更次数 >= 3
- **组合条件**: H9 AND H10 (不是 H9 OR H10)

## 输入数据格式 / Input Data Format

```python
required_columns = [
    'Date',          # 日期
    'S_YR',          # 年份
    'OFFICE',        # 办公室
    'AMGR',          # AMGR标识
    'LP',            # LP标识
    'RANK_x',        # 职级 (10=LP, 20=AMGR)
    'SASHIHIKI',     # 薪酬
    'STATUS',        # 状态 ('A'=在职, 'T'=离职)
    'SHOBUN'         # 纪律处分记录
]
```

## 输出结果 / Output Results

### 验证结果表 / Validation Results Table
```python
results_columns = [
    'OFFICE',                        # 办公室
    'YEAR',                         # 年份
    'H9_CONDITION_MET',             # H9条件是否满足
    'H10_CONDITION_MET',            # H10条件是否满足
    'H9_AND_H10_CONDITION',         # 组合条件
    'NEXT_YEAR_DISCIPLINARY_ACTION' # 次年纪律处分
]
```

### 性能指标 / Performance Metrics
- 混淆矩阵 (Confusion Matrix)
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1 Score)

## 使用方法 / Usage

### 基本验证 / Basic Validation
```python
from hypothesis_validator_9_10_combined import Hypothesis9And10CombinedValidator

# 初始化验证器
validator = Hypothesis9And10CombinedValidator()

# 执行验证
results = validator.validate_combined_hypothesis(dataframe)

# 打印结果
validator.print_analysis_summary()
```

### 机器学习分析 / ML Analysis
```python
from ml_analysis import MLAnalyzer

# 初始化ML分析器
ml_analyzer = MLAnalyzer()

# 准备特征
X, y = ml_analyzer.prepare_features(validator.results)

# 训练模型
cv_results = ml_analyzer.train_xgboost_model()

# 特征重要性分析
feature_imp, shap_imp, shap_values = ml_analyzer.analyze_feature_importance()
```

### 运行测试 / Run Tests
```bash
cd hypothesis_testing/hypothesis_9_10_combined
python test_hypothesis_9_10_combined.py
```

## 测试数据特征 / Test Data Characteristics

- **办公室数量**: 20个
- **年份范围**: 2019-2023 (5年)
- **高变更办公室**: 前5个办公室模拟高AMGR流动性
- **问题LP**: 高变更办公室中的部分LP模拟低薪酬+早期离职
- **纪律处分**: 与组合条件相关的概率性生成

## 预期结果 / Expected Results

基于测试数据设计，预期：
- H9条件满足的案例主要集中在高变更办公室
- H10条件满足的案例也主要在高变更办公室
- 组合条件(H9 AND H10)的案例数量较少但更有预测性
- 次年纪律处分与组合条件应显示正相关关系
