# Hypothesis 10: AMGR Turnover and Disciplinary Actions Analysis

## 概要 (Overview)

Hypothesis 10 分析営業所長(AMGR)的高频变更是否能预测次年的纪律处分。

**仮説**: if "(变数1) = 1 & 变数2(t)" then (变数3) = 1

### 變数定義 (Variable Definitions)

- **变数1(t)**: "営業所長(AMGR)が1年以内" (AMGR Within 1 Year)
  - 营业所长在1年内的时间窗口分析

- **变数2(t)**: "3回以上変わる" (Changes 3 or More Times)
  - AMGR变更次数>=3次（修正后：新增+离开的总变更数）
  - 识别高频人事变动的办公室

- **变数3(t)**: "オフィス(支社)の翌年の懲戒処分の有無" (Disciplinary Actions in Following Year)
  - 办公室次年是否发生纪律处分（二元值）
  - 1: 有纪律处分, 0: 无纪律处分

## 分析ロジック (Analysis Logic)

### 1. 核心假设逻辑

```
if (办公室年度内AMGR变更次数) >= 3:
    then 该办公室次年有纪律处分 = 1
```

**业务解释**:
- 当办公室的营业所长频繁变更时（>=3次）
- 反映该办公室的管理不稳定和组织问题
- 这种管理混乱预示着次年纪律处分的发生

### 2. 修正后的变更检测逻辑

**❌ 原始错误逻辑**:
```python
# 只检测新增AMGR，忽略离开的AMGR
amgr_change_count = len(set(current_amgrs) - set(prev_amgrs))
```

**✅ 修正后的正确逻辑**:
```python
# 计算总变更次数：新增 + 离开
new_amgrs = set(current_amgrs) - set(prev_amgrs)      # 新增的AMGR
left_amgrs = set(prev_amgrs) - set(current_amgrs)     # 离开的AMGR
total_changes = len(new_amgrs) + len(left_amgrs)      # 总变更次数
is_high_change = total_changes >= 3
```

### 3. 数据处理流程

1. **办公室年度分组**: 按OFFICE和S_YR分组分析
2. **AMGR变更检测**: 计算每年的AMGR总变更次数
3. **高变更标识**: 变更次数>=3次标记为高变更年
4. **次年纪律处分**: 使用shift(-1)获取次年SHOBUN数据
5. **假设验证**: 验证高变更年是否预测次年纪律处分

### 4. 两种验证模式

#### 模式1: XGBoost机器学习验证
- **特征**: AMGR高变更标志（二元值）
- **目标**: 次年纪律处分标志
- **方法**: XGBoost分类器 + SMOTE平衡
- **评估**: 准确率、F1分数、混淆矩阵、SHAP分析

#### 模式2: 简单验证模式
- **方法**: 直接二元分类验证
- **逻辑**: AMGR变更>=3次 → 次年纪律处分标志
- **评估**: 混淆矩阵、F1分数、准确率

### 5. 统计分析

- **卡方检验**: AMGR高变更标志 vs 次年纪律处分标志
- **t检验**: 高变更组vs低变更组的次年纪律处分数量比较
- **相关性分析**: AMGR变更与次年纪律处分的相关性

## ファイル構成 (File Structure)

```
hypothesis_testing/hypothesis_10/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_10.py     # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_10/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_10 \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_10/run.sh
./hypothesis_testing/hypothesis_10/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_10 \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, OFFICE, RANK_x, JOB_YYY, JOB_MM 列が必要

2. **懲戒処分_事故区分等追加_hashed.csv** (重要)
   - LP NO, SHOBUN 列が必要
   - 次年纪律处分分析の核心

3. **時系列データ** (重要)
   - 年度別AMGR変更検出に必須
   - S_YR列による年度分析

### データ要件の詳細

- **OFFICE情報**: 办公室レベル分析に必須
- **AMGR情報**: 営業所長変更検出に必須
- **年度データ**: 次年予測分析に必須
- **纪律处分データ**: 目标变量分析に必須

## 出力結果 (Output Results)

### hypothesis_10_results.json

```json
{
  "hypothesis_id": "H10_amgr_turnover",
  "data_shape": [行数, 列数],
  "descriptive_statistics": {
    "amgr_changes": {
      "total_office_years": "総办公室年度記録数",
      "high_change_office_years": "高変更办公室年度数",
      "high_change_rate": "高変更率",
      "average_total_changes": "平均総変更数",
      "max_total_changes": "最大変更数"
    },
    "next_year_disciplinary": {
      "office_years_with_next_disciplinary": "次年処分ありの办公室年度数",
      "next_year_disciplinary_rate": "次年処分率",
      "average_next_year_shobun": "次年平均処分数"
    },
    "office_lps": {
      "average_lps_per_office_year": "办公室年度平均LP数",
      "total_unique_offices": "総办公室数"
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
      "group_0_mean": 低変更群平均,
      "group_1_mean": 高変更群平均,
      "t_statistic": "t統計量",
      "p_value": "p値",
      "significant": 統計的有意性
    },
    "correlation": {
      "amgr_turnover_vs_next_year_disciplinary": 相関係数
    }
  },
  "amgr_turnover_analysis": {
    "total_office_year_records": 総办公室年度記録数,
    "offices_with_high_amgr_turnover": 高AMGR変更の办公室年度数,
    "offices_with_next_year_disciplinary": 次年処分ありの办公室年度数,
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
- **明確な相関**: AMGR変更と次年処分の正の相関

### 仮説支持率の計算

```
支持率 = (高AMGR変更 かつ 次年処分あり) / (高AMGR変更)
```

### ビジネス洞察

1. **管理安定性指標**: AMGR変更頻度による組織安定性評価
2. **早期警告システム**: 高頻度人事変動による次年リスク予測
3. **組織健全性**: 営業所長変更パターンと規律問題の関連性
4. **予防的管理**: 人事変動による将来的問題の事前察知

## 技術的詳細 (Technical Details)

### 依存関係

- pandas: データ処理
- numpy: 数値計算
- scikit-learn: 機械学習
- xgboost: 勾配ブースティング (オプション)
- imbalanced-learn: クラス平衡 (オプション)
- shap: 特徴量解釈 (オプション)
- scipy: 統計検定

### 修正されたアルゴリズム

1. **修正変更検出**: `new_amgrs + left_amgrs = total_changes`
2. **年度分析**: groupby(['OFFICE', 'S_YR'])による年度別分析
3. **次年予測**: shift(-1)による次年データ取得
4. **二重検証**: XGBoost + 簡単分類の組み合わせ

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **AMGR変更検出エラー**
   ```bash
   Warning: AMGR change detection logic corrected
   ```

2. **次年データ不足**
   ```bash
   Warning: Dropping records without next year data
   ```

3. **办公室データ不完整**
   - OFFICE列の存在確認
   - 年度データの連続性確認

### データ品質チェック

- OFFICE列の存在と妥当性確認
- AMGR情報の完整性確認
- 年度データの連続性確認
- SHOBUN情報の妥当性確認

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - 修正されたAMGR変更検出ロジック
  - 次年纪律処分予測機能
  - 办公室レベル年度分析
  - 二重検証モード実装
