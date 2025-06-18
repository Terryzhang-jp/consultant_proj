# Hypothesis 9: Low Salary and Turnover Analysis

## 概要 (Overview)

Hypothesis 9 は低薪酬LP的连续低收入期间和后续离职是否能预测公司的纪律处分。

**仮説**: if "(变数3) >= 1" then (变数4) = 1

### 変数定義 (Variable Definitions)

- **变数1(t)**: "低報酬(SASHIHIKI20万円未満)が6か月継続" (Low Salary Consecutive Period)
  - SASHIHIKI < 200,000日元连续6个月（二元值）
  - 识别持续低收入的LP

- **变数2(t)**: "LPが2年以内に退職" (LP Turnover Within 2 Years)
  - LP在低薪酬期开始后2年内离职（二元值）
  - 基于STATUS='T'判定离职

- **变数3(t)**: 变数1(t) & 变数2(t) LP 総人数 (Total LPs Meeting Both Conditions)
  - 同时满足变数1和变数2的LP总人数（连续值）
  - 营业所级别的聚合统计

- **变数4(t)**: その会社での懲戒処分の有無 (Disciplinary Actions Presence)
  - 该营业所是否发生纪律处分（二元值）
  - 1: 有纪律处分, 0: 无纪律处分

## 分析ロジック (Analysis Logic)

### 1. 核心假设逻辑

```
if (营业所内低薪离职LP数量) >= 1:
    then 该营业所有纪律处分 = 1
```

**业务解释**:
- 当营业所有LP因持续低薪酬而离职时
- 反映该营业所的薪酬管理和人才保留问题
- 这种状况预示着管理问题和纪律处分的发生

### 2. 数据处理流程

1. **低薪酬检测**: SASHIHIKI < 200,000日元
2. **连续性判定**: 使用rolling window检测连续6个月
3. **离职判定**: 基于STATUS='T'确定离职日期
4. **时间窗口**: 从低薪酬开始到离职≤730天（2年）
5. **营业所聚合**: 按AMGR分组统计符合条件的LP数量
6. **纪律处分关联**: 计算营业所级别的纪律处分发生率

### 3. 两种验证模式

#### 模式1: XGBoost机器学习验证
- **特征**: 低薪离职LP数量（连续值）
- **目标**: 纪律处分发生标志
- **方法**: XGBoost分类器 + SMOTE平衡
- **评估**: 准确率、F1分数、混淆矩阵、SHAP分析

#### 模式2: 简单验证模式
- **方法**: 直接二元分类验证
- **逻辑**: 低薪离职LP数量 >= 1 → 纪律处分标志
- **评估**: 混淆矩阵、F1分数、准确率

### 4. 统计分析

- **卡方检验**: 低薪离职标志 vs 纪律处分标志
- **t检验**: 有无纪律处分组的低薪离职数量比较
- **相关性分析**: 低薪离职数量与纪律处分的相关性

## ファイル構成 (File Structure)

```
hypothesis_testing/hypothesis_9/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_9.py      # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_9/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_9 \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_9/run.sh
./hypothesis_testing/hypothesis_9/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_9 \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, RANK_x, JOB_YYY, JOB_MM 列が必要

2. **報酬データ_hashed.csv** (重要)
   - SYAIN_CODE, S_YR, S_MO, SASHIHIKI 列が必要
   - SASHIHIKI列は薪酬分析の核心

3. **懲戒処分_事故区分等追加_hashed.csv**
   - LP NO, SHOBUN 列が必要

4. **STATUS情報** (重要)
   - 離職判定に必要
   - STATUS='T'で離職を判定

### データ要件の詳細

- **時系列データ**: 連続6個月の低薪酬検出に必須
- **AMGR情報**: 営業所レベル分析に必須
- **薪酬データ**: 200,000円閾値判定に必須
- **離職データ**: 2年以内離職判定に必須

## 出力結果 (Output Results)

### hypothesis_9_results.json

```json
{
  "hypothesis_id": "H9_low_salary_turnover",
  "data_shape": [行数, 列数],
  "descriptive_statistics": {
    "low_salary_turnover": {
      "mean": "営業所平均低薪離職数",
      "count_with_turnover": "離職ありの営業所数",
      "rate_with_turnover": "離職率"
    },
    "disciplinary_actions": {
      "total_amgr": "総営業所数",
      "with_disciplinary_actions": "処分ありの営業所数",
      "disciplinary_action_rate": "処分率"
    },
    "total_lps": {
      "mean_lps_per_amgr": "営業所平均LP数",
      "total_lps_all_amgr": "全LP総数"
    }
  },
  "xgboost_validation": {
    "accuracy": "XGBoost精度",
    "f1_score": "F1スコア",
    "confusion_matrix": 混淆行列,
    "feature_importance": 特徴重要度,
    "shap_analysis": "SHAP分析結果"
  },
  "simple_validation": {
    "accuracy": 簡単検証精度,
    "f1_score": "F1スコア",
    "confusion_matrix": 混淆行列,
    "method": "検証方法説明"
  },
  "statistical_analysis": {
    "chi_square_test": {
      "chi2_statistic": カイ二乗統計量,
      "p_value": "p値",
      "significant": 統計的有意性,
      "contingency_table": 分割表
    },
    "t_test": {
      "group_0_mean": 処分なし群平均,
      "group_1_mean": 処分あり群平均,
      "t_statistic": "t統計量",
      "p_value": "p値",
      "significant": 統計的有意性
    },
    "correlation": {
      "low_salary_turnover_vs_disciplinary": 相関係数
    }
  },
  "low_salary_turnover_analysis": {
    "total_amgr_records": 総営業所記録数,
    "amgr_with_low_salary_turnover": 低薪離職ありの営業所数,
    "amgr_with_disciplinary_actions": 処分ありの営業所数,
    "hypothesis_support_rate": 仮説支持率
  },
  "conclusion": "結論文"
}
```

## 解釈ガイド (Interpretation Guide)

### 成功指標

- **高い仮説支持率**: Support Rate > 0.6
- **統計的有意性**: p値 < 0.05 (カイ二乗検定、t検定)
- **機械学習性能**: XGBoost F1-score > 0.6
- **明確な相関**: 低薪離職数と処分の正の相関

### 仮説支持率の計算

```
支持率 = (低薪離職>=1 かつ 処分あり) / (低薪離職>=1)
```

### ビジネス洞察

1. **薪酬管理指標**: 200,000円閾値による低薪酬LP識別
2. **人材保留**: 連続低薪酬期間の離職リスク分析
3. **早期警告システム**: 低薪離職による処分リスク予測
4. **営業所管理**: AMGR単位での人事管理評価

## 技術的詳細 (Technical Details)

### 依存関係

- pandas: データ処理
- numpy: 数値計算
- scikit-learn: 機械学習
- xgboost: 勾配ブースティング (オプション)
- imbalanced-learn: クラス平衡 (オプション)
- shap: 特徴量解釈 (オプション)
- scipy: 統計検定

### アルゴリズム詳細

1. **連続低薪酬検出**: rolling(window=6, min_periods=6) による6か月連続判定
2. **離職日計算**: STATUS='T'による離職日特定
3. **2年窓口**: (leave_date - streak_start_date).days <= 730
4. **営業所集計**: groupby('AMGR')による営業所単位統計
5. **二重検証**: XGBoost + 簡単分類の組み合わせ

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **STATUS列なし**
   ```bash
   Warning: Creating dummy STATUS data for turnover analysis
   ```

2. **SASHIHIKI データ不足**
   ```bash
   Error: SASHIHIKI column required for salary analysis
   ```

3. **時系列データ不完整**
   - 連続6か月検出に影響
   - データ期間を確認

### データ品質チェック

- SASHIHIKI列の存在と数値妥当性確認
- 時系列データの連続性確認
- STATUS情報の完整性確認
- AMGR情報の妥当性確認

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - 連続低薪酬検出機能
  - 2年離職分析機能
  - 営業所レベル集計
  - 二重検証モード実装
