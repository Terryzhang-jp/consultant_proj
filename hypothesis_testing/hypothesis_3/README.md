# Hypothesis 3: AMGR 历史纪律处分影响分析

## 📋 假设定义

**假设**: `if "(变数1) == 1" then "(变数2) > 0"`

- **变数1**: AMGRのLP時代の懲戒処分の有無(0:処分なし、1:処分あり)
- **变数2**: このAMGRが管理するLPの懲戒処分の有無

### 业务含义
如果一个AMGR在他还是LP时代有过纪律处分，那么他管理的LP也更可能受到纪律处分。这个假设探讨了管理者的历史行为对其团队管理效果的影响。

---

## 🏗️ 代码结构

```
hypothesis_3/
├── __init__.py                  # 模块初始化
├── hypothesis_validator_3.py    # 主要验证器类
├── run.py                      # Python运行脚本
├── run.sh                      # Shell运行脚本
└── README.md                   # 本文档
```

### 核心类和方法

#### `Hypothesis3Validator`
- **主要方法**: `validate_hypothesis_3_amgr_influence()`
- **核心逻辑**: `_analyze_amgr_shobun_influence()`
- **历史分析**: `_mark_amgr_shobun_history()`
- **违规计算**: `_calculate_amgr_lp_violations()`
- **统计分析**: `_perform_hypothesis_3_statistical_test()`

---

## 📥 输入 (Input)

### 数据要求
```python
final_dataset = pd.DataFrame({
    'LP': ['LP_0001', 'LP_0002', ...],           # Life Planner ID
    'AMGR': ['LP_1001', 'LP_1002', ...],         # Area Manager ID
    'RANK': [10, 20, 10, ...],                   # 员工等级 (10=LP, 20=AMGR)
    'SHOBUN': ['警告', '注意', NaN, ...],         # 纪律处分
    'S_YR': [2020, 2021, ...],                   # 年份
    'S_MO': [4, 5, 6, ...],                      # 月份
})
```

### 数据过滤条件
- **RANK要求**: 需要同时有RANK=10（LP）和RANK=20（AMGR）的数据
- **历史追踪**: 需要能够追踪AMGR从LP到AMGR的职业发展轨迹
- **必要列**: LP, AMGR, RANK, SHOBUN, S_YR

---

## 📤 输出 (Output)

### 主要输出结构
```python
{
    'hypothesis_id': 'H3_amgr_influence',
    'data_shape': [50, 8],                      # 处理后数据形状 (AMGR记录)
    
    'amgr_influence_analysis': {
        'total_amgrs': 50,                      # 总AMGR数量
        'amgrs_with_history': 15,               # 有纪律处分历史的AMGR数
        'amgrs_with_lp_violations': 25,         # 有LP违规的AMGR数
        'both_conditions_met': 8,               # 两个条件都满足的AMGR数
        'prediction_accuracy': {
            'accuracy': 0.720,                  # 整体准确性
            'hypothesis_accuracy': 0.533,      # 假设准确性
            'precision': 0.320,                 # 精确度
            'recall': 0.800                     # 召回率
        }
    },
    
    'statistical_test': {
        'test_type': 'Hypothesis 3 Validation: if AMGR_Has_SHOBUN_History = 1 then Shobun_Flag = 1',
        'confusion_matrix': [[20, 5], [17, 8]], # 混淆矩阵
        'f1_score': 0.421,                      # F1分数
        'condition_true_cases': 15,             # 有历史的AMGR数
        'hypothesis_support_rate': 0.533        # 假设支持率
    },
    
    'amgr_patterns': {
        'amgr_distribution': {0: 35, 1: 15},    # AMGR历史分布
        'lp_violation_distribution': {0: 25, 1: 25}, # LP违规分布
        'cross_tabulation': {...}               # 交叉表分析
    },
    
    'conclusion': 'Hypothesis 3 Results: 15 AMGRs with disciplinary history...'
}
```

---

## ⚙️ 中间处理逻辑

### 步骤1: AMGR历史纪律处分标记
```python
# 1. 获取所有AMGR列表 (RANK=20)
amgr_list = df[df['RANK'] == 20]['LP'].unique()

# 2. 检查每个AMGR在LP时代是否有过SHOBUN
had_shobun = df[
    (df['LP'] == amgr) &
    (df['RANK'] == 10) &  # 当他们是LP时
    (df['SHOBUN'].notna())  # 有纪律处分
].shape[0] > 0

# 3. 创建历史标记
Has_SHOBUN_History = int(had_shobun)
```

### 步骤2: LP违规统计
```python
# 4. 按AMGR和财政年度分组
for amgr in amgrs:
    for fy in fiscal_years:
        # 5. 计算该AMGR管理的LP总数
        total_lps = amgr_fy_data[amgr_fy_data['RANK'] == 10]['LP'].nunique()
        
        # 6. 计算纪律处分数量
        shobun_count = amgr_fy_data[amgr_fy_data['RANK'] == 10]['SHOBUN'].notna().sum()
```

### 步骤3: 总体统计汇总
```python
# 7. 按AMGR汇总所有年度数据
amgr_total_stats = amgr_lp_violations_df.groupby('AMGR').agg({
    'Total_LPs': 'sum',
    'Shobun_Count': 'sum',
    'AMGR_Has_SHOBUN_History': 'first'
})

# 8. 计算比率和标志
Shobun_per_LP = Total_Shobun_Count / Total_LPs_All_Years
Shobun_Flag = (Shobun_per_LP > 0).astype(int)
```

### 步骤4: 假设验证
```python
# 9. 验证假设: AMGR_Has_SHOBUN_History=1 时，Shobun_Flag 是否也=1
hypothesis_validation = validate(AMGR_Has_SHOBUN_History, Shobun_Flag)
```

---

## 🔍 关键特性

### AMGR历史追踪
- **职业轨迹**: 追踪个人从LP (RANK=10) 到AMGR (RANK=20) 的发展
- **历史标记**: 识别AMGR在LP时代的纪律处分记录
- **时间维度**: 考虑历史行为对当前管理效果的影响

### 管理效果分析
- **团队违规率**: 计算每个AMGR管理的LP的纪律处分率
- **影响评估**: 评估AMGR历史对团队表现的预测能力
- **模式识别**: 识别管理者行为模式的传承效应

### 统计验证
- **混淆矩阵**: 评估预测性能
- **假设支持率**: 条件满足时结果发生的比例
- **交叉表分析**: 详细的分类统计

---

## 🚀 使用示例

```python
from hypothesis_testing.hypothesis_3 import Hypothesis3Validator

# 初始化验证器
validator = Hypothesis3Validator()

# 运行假设验证
results = validator.validate_hypothesis_3_amgr_influence(final_dataset)

# 查看结果
print(f"Total AMGRs: {results['amgr_influence_analysis']['total_amgrs']}")
print(f"AMGRs with history: {results['amgr_influence_analysis']['amgrs_with_history']}")
print(f"Hypothesis accuracy: {results['amgr_influence_analysis']['prediction_accuracy']['hypothesis_accuracy']}")

# 查看AMGR模式
patterns = results['amgr_patterns']
print(f"AMGR distribution: {patterns['amgr_distribution']}")
```

### 命令行使用
```bash
# Python脚本
python hypothesis_testing/hypothesis_3/run.py --data-path test_data_folder --output-path hypothesis_testing/hypothesis_3

# Shell脚本
cd hypothesis_testing/hypothesis_3
./run.sh [DATA_PATH] [OUTPUT_PATH]
```

---

## 📊 业务价值

1. **管理者选拔**: 识别可能影响团队表现的管理者特征
2. **风险预警**: 预测哪些团队可能出现更多纪律问题
3. **培训需求**: 为有历史问题的管理者提供针对性培训
4. **组织文化**: 理解行为模式在组织中的传承机制

---

## 🔬 分析维度

### 历史影响指标
- **AMGR_Has_SHOBUN_History**: AMGR在LP时代的纪律处分历史
- **Shobun_per_LP**: 每个LP的平均纪律处分率
- **Total_LPs_All_Years**: AMGR管理的LP总数
- **Total_Shobun_Count**: 管理团队的总纪律处分数

### 管理效果评估
- **假设支持率**: 有历史的AMGR中，其团队确实有违规的比例
- **预测准确性**: 基于AMGR历史预测团队表现的准确度
- **影响强度**: 历史行为对当前管理效果的影响程度

---

## ⚠️ 注意事项

1. **数据完整性**: 需要完整的职业发展轨迹数据
2. **时间对齐**: 确保历史数据和当前管理数据的时间一致性
3. **样本大小**: 需要足够的AMGR样本以获得统计显著性
4. **因果关系**: 注意区分相关性和因果关系
5. **隐私保护**: 处理个人历史数据时需要注意隐私保护

---

## 🔄 扩展可能性

1. **时间窗口**: 可以调整历史追溯的时间范围
2. **严重程度**: 考虑不同类型纪律处分的严重程度权重
3. **多维分析**: 结合其他绩效指标进行综合分析
4. **预测模型**: 基于历史模式构建更复杂的预测模型
5. **干预策略**: 基于分析结果制定管理干预策略
