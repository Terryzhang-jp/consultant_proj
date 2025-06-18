# Hypothesis 7: Meeting Attendance Difference Analysis

## 概要 (Overview)

Hypothesis 7 は高薪LP和低薪LP的会议出席率差异是否能预测下一年度的纪律处分。

**仮説**: if "(变数3) < 0" then (变数4) = 1

### 変数定義 (Variable Definitions)

- **变数1(t)**: "高報酬LPのMTG出席率" (High-salary LP Meeting Attendance Rate)
  - 各支社内高薪LP的会议出席率（连续值）
  - 基于中位数SASHIHIKI划分的高薪组

- **变数2(t)**: "低報酬LPのMTG出席率" (Low-salary LP Meeting Attendance Rate)
  - 各支社内低薪LP的会议出席率（连续值）
  - 基于中位数SASHIHIKI划分的低薪组

- **变数3(t)**: 变数1(t) - 变数2(t) (Attendance Difference)
  - 高薪LP出席率 - 低薪LP出席率（连续值）
  - 负值表示低薪LP出席率更高

- **变数4(t)**: 営業所の翌年度の懲戒処分の有無 (Disciplinary Actions in Following Year)
  - 支社下一年度是否发生纪律处分（二元值）
  - 1: 有纪律处分, 0: 无纪律处分

## 分析ロジック (Analysis Logic)

### 1. 数据准备流程

1. **支社定义**: 以AMGR为单位定义各支社
2. **薪资分组**: 基于各支社年度内SASHIHIKI中位数分为高薪/低薪LP
3. **出席率计算**: 分别计算高薪LP和低薪LP的会议出席率
4. **差值计算**: 计算出席率差异 (高薪 - 低薪)
5. **时间偏移**: 将纪律处分数据向前偏移一年实现"翌年度"预测

### 2. 核心假设逻辑

```
if (高薪LP出席率 - 低薪LP出席率) < 0:
    then 翌年度有纪律处分 = 1
```

**业务解释**:
- 当低薪LP比高薪LP更积极参与会议时
- 可能反映组织内部的不平衡或紧张关系
- 这种状况预示着下一年度可能出现纪律问题

### 3. 两种验证模式

#### 模式1: XGBoost机器学习验证
- **特征**: 出席率差异（连续值）
- **目标**: 翌年度纪律处分标志
- **方法**: XGBoost分类器 + SMOTE平衡
- **评估**: 准确率、F1分数、混淆矩阵、SHAP分析

#### 模式2: 简单验证模式
- **方法**: 直接二元分类验证
- **逻辑**: 出席率差异 < 0 → 纪律处分标志
- **评估**: 混淆矩阵、F1分数、准确率

### 4. 统计分析

- **卡方检验**: 出席率差异标志 vs 纪律处分标志
- **t检验**: 有无纪律处分组的出席率差异比较
- **相关性分析**: 出席率差异与纪律处分的相关性

## ファイル構成 (File Structure)

```
hypothesis_testing/hypothesis_7/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_7.py      # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_7/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_7 \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_7/run.sh
./hypothesis_testing/hypothesis_7/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_7 \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, RANK_x, JOB_YYY, JOB_MM, コンプライアンス 列が必要

2. **報酬データ_hashed.csv**
   - SYAIN_CODE, S_YR, S_MO, SASHIHIKI 列が必要
   - SASHIHIKI用于薪资分组

3. **懲戒処分_事故区分等追加_hashed.csv**
   - LP NO, SHOBUN 列が必要

4. **MTG出席率データ** (重要)
   - LP, S_YR, S_MO, MTG_ATTENDANCE 列が必要
   - または '出欠' 列が必要
   - ファイル名: MTG出席率2021-2023_hashed.csv など

5. **苦情データ_hashed.xlsx** (オプション)
   - LP, 苦情 列が必要

### データ要件の詳細

- **AMGR情報**: 支社定義に必須
- **SASHIHIKI**: 高薪/低薪分組に必須
- **会議出席データ**: 核心分析対象
- **時系列データ**: 複数年度のデータが必要（時間偏移のため）

## 出力結果 (Output Results)

### hypothesis_7_results.json

```json
{
  "hypothesis_id": "H7_attendance_difference",
  "data_shape": [行数, 列数],
  "descriptive_statistics": {
    "attendance_difference": {
      "mean": 平均差値,
      "median": 中央差値,
      "negative_count": 負値件数,
      "negative_rate": 負値率
    },
    "high_salary_attendance": {
      "mean": 高薪LP平均出席率,
      "std": 標準偏差
    },
    "low_salary_attendance": {
      "mean": 低薪LP平均出席率,
      "std": 標準偏差
    },
    "disciplinary_actions": {
      "total_records": 総記録数,
      "with_disciplinary_actions": 処分ありの数,
      "disciplinary_action_rate": 処分率
    }
  },
  "xgboost_validation": {
    "accuracy": XGBoost精度,
    "f1_score": "F1スコア",
    "confusion_matrix": 混淆行列,
    "feature_importance": 特徴重要度,
    "shap_analysis": SHAP分析結果
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
      "attendance_difference_vs_disciplinary": 相関係数
    }
  },
  "attendance_difference_analysis": {
    "total_records": 総支社年記録数,
    "negative_difference_count": 負差値件数,
    "negative_difference_rate": 負差値率,
    "disciplinary_actions_count": 処分件数,
    "disciplinary_action_rate": 処分率,
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
- **明確な差異**: 出席率差異の分布が明確

### 仮説支持率の計算

```
支持率 = (出席率差異<0 かつ 翌年処分あり) / (出席率差異<0)
```

### ビジネス洞察

1. **組織健全性指標**: 出席率差異が組織の健全性を反映
2. **早期警告システム**: 負の差異による翌年リスク予測
3. **管理改善**: 薪資公平性と参与積極性のバランス
4. **予防的介入**: 差異検出時の早期対応策

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

1. **薪資分組**: 各支社年度内SASHIHIKI中位数基準
2. **出席率計算**: 薪資グループ別平均出席率
3. **時間偏移**: groupby('AMGR').shift(-1) による翌年予測
4. **二重検証**: XGBoost + 簡単分類の組み合わせ

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **出席データなし**
   ```bash
   # ダミーデータで実行
   Warning: No attendance column found, using dummy data
   ```

2. **AMGR情報不足**
   ```bash
   ValueError: No data with AMGR information found
   ```

3. **時系列データ不足**
   - 複数年度のデータが必要
   - 時間偏移のため最低2年分必要

### データ品質チェック

- AMGR列の完整性確認
- SASHIHIKI値の妥当性確認
- 出席データの存在確認
- 年度データの連続性確認

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - 二重検証モード実装
  - XGBoost + 簡単分類
  - 統計分析機能
  - 時間偏移予測機能
