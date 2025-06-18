# consultant_proj

## 项目概述 / Project Overview

这是一个用于假设验证和统计分析的咨询项目，包含多个假设的验证框架和统一的数据处理管道。项目采用模块化设计，支持灵活的数据加载和假设验证。

This is a consulting project for hypothesis validation and statistical analysis, featuring multiple hypothesis validation frameworks and a unified data processing pipeline. The project uses modular design to support flexible data loading and hypothesis validation.

## 项目结构 / Project Structure

```
consultant_proj/
├── README.md                           # 项目总体说明
├── requirements.txt                    # Python依赖包列表
├── data/                              # 数据文件夹
│   ├── generate_comprehensive_test_data.py  # 测试数据生成脚本
│   └── *.csv, *.xlsx                  # 数据文件
├── data_loading/                       # 数据加载模块
│   └── unified_data_loader.py         # 统一数据加载器
├── hypothesis_testing/                 # 假设验证主目录
│   ├── hypothesis_1/                   # 假设1: LP持续低业绩预测懲戒処分
│   │   ├── hypothesis_1.py            # 假设1验证脚本
│   │   └── README.md                  # 假设1说明文档
│   ├── hypothesis_2/                   # 假设2: LP收入集中度预测懲戒処分
│   │   ├── hypothesis_2.py            # 假设2验证脚本
│   │   └── README.md                  # 假设2说明文档
│   ├── hypothesis_3/                   # 假设3: AMGR历史懲戒処分影响
│   │   ├── hypothesis_3.py            # 假设3验证脚本
│   │   └── README.md                  # 假设3说明文档
│   └── ...                            # 其他假设
├── test_hypothesis_1.py               # 假设1测试脚本
├── test_hypothesis_2.py               # 假设2测试脚本
├── test_hypothesis_3.py               # 假设3测试脚本
└── run_all_hypotheses.py              # 运行所有假设的脚本
```

## 假设列表 / Hypothesis List

| 假设编号 | 假设描述 | 状态 |
|---------|---------|------|
| Hypothesis 1 | LP持续低业绩预测懲戒処分 | ✅ 完成 |
| Hypothesis 2 | LP收入集中度预测懲戒処分 | ✅ 完成 |
| Hypothesis 3 | AMGR历史懲戒処分影响LP懲戒処分 | ✅ 完成 |
| Hypothesis 4 | 待实现 | ⏳ 开发中 |
| Hypothesis 5 | 待实现 | ⏳ 开发中 |
| Hypothesis 6 | 待实现 | ⏳ 开发中 |
| Hypothesis 7 | 待实现 | ⏳ 开发中 |
| Hypothesis 8 | 待实现 | ⏳ 开发中 |
| Hypothesis 9 | 待实现 | ⏳ 开发中 |
| Hypothesis 10 | 待实现 | ⏳ 开发中 |

## 环境设置 / Environment Setup

### 1. 创建虚拟环境 / Create Virtual Environment

```bash
# 运行环境设置脚本
./setup_environment.sh

# 或手动创建
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 激活环境 / Activate Environment

```bash
source venv/bin/activate
```

## 使用方法 / Usage

### 运行单个假设验证 / Run Single Hypothesis

```bash
cd hypothesis_testing/hypothesis_X
python hypothesis_X.py
```

### 运行组合假设验证 / Run Combined Hypothesis

```bash
cd hypothesis_testing/hypothesis_9_10_combined
python hypothesis_9_10_combined.py
```

## 数据要求 / Data Requirements

每个假设验证脚本都需要特定格式的输入数据。详细的数据格式要求请参考各个假设文件夹中的README.md文件。

Each hypothesis validation script requires input data in specific formats. For detailed data format requirements, please refer to the README.md file in each hypothesis folder.

### 通用数据字段 / Common Data Fields

- `Date`: 日期字段
- `S_YR`: 年份
- `OFFICE`: 办公室标识
- `AMGR`: AMGR标识
- `LP`: LP标识
- `RANK_x`: 职级
- `SASHIHIKI`: 薪酬数据
- `STATUS`: 状态信息
- `SHOBUN`: 纪律处分记录

## 技术栈 / Technology Stack

- **Python 3.8+**: 主要编程语言
- **pandas**: 数据处理和分析
- **numpy**: 数值计算
- **scikit-learn**: 机器学习
- **xgboost**: 梯度提升算法
- **shap**: 模型解释性分析
- **matplotlib/seaborn**: 数据可视化
- **statsmodels**: 统计分析

## 贡献指南 / Contributing

1. 每个假设验证模块都应该是独立的
2. 代码应该包含充分的注释和文档
3. 所有假设都应该有对应的README.md说明文档
4. 遵循Python PEP 8编码规范

## 许可证 / License

本项目仅供内部研究使用。

This project is for internal research use only.

## 联系信息 / Contact

如有问题或建议，请联系项目维护团队。

For questions or suggestions, please contact the project maintenance team.
# consultant_proj
