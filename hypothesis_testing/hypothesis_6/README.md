# Hypothesis 6: Multi-dimensional Risk Factor Analysis

## 概要 (Overview)

Hypothesis 6 は4つの連続値リスク要因の組み合わせが懲戒処分を予測できるかを分析します。

**仮説**: if "(变数1, 连续值)" & (变数2, 连续值) & (变数3, 连续值) & (变数4, 连续值), then (变数5)=1

### 変数定義 (Variable Definitions)

- **变数1**: MTG出席率 (Meeting Attendance Rate)
  - 会議への出席率（連続値）
  - 管理者の積極性と責任感を示す指標

- **变数2**: 苦情率 (Complaint Rate)
  - 顧客からの苦情発生率（連続値）
  - サービス品質と顧客満足度を示す指標

- **变数3**: コンプライアンス違反疑義 (Compliance Violation Suspicion)
  - コンプライアンス違反の疑いがある事案の発生率（連続値）
  - 法令遵守意識と内部統制の有効性を示す指標

- **变数4**: 事務ミス (Administrative Errors)
  - 事務処理におけるミスの発生率（連続値）
  - 業務精度と注意力を示す指標

- **变数5**: 懲戒処分が出る (Disciplinary Action Occurs)
  - 翌年度に懲戒処分が発生するかどうか（バイナリ）
  - 1: 懲戒処分あり, 0: 懲戒処分なし

## 分析ロジック (Analysis Logic)

### 1. データ準備 (Data Preparation)

1. **会計年度の定義**
   - 4月から翌年3月までを1会計年度として定義
   - 各指標を会計年度ベースで集計

2. **AMGR単位での年次指標計算**
   - 各AMGRについて年度ごとの4つのリスク指標を計算
   - LP（RANK=10）のデータのみを対象として集計

3. **時間オフセット予測**
   - 当年度の4指標を使用して翌年度の懲戒処分を予測
   - 予防的リスク管理の観点から先行指標として活用

### 2. 4つのリスク指標の計算

#### MTG出席率 (Meeting Attendance Rate)
```python
mtg_attendance_rate = mtg_attendance_count / total_lp_months
```

#### 苦情率 (Complaint Rate)
```python
complaint_rate = complaint_count / total_lp_months
```

#### コンプライアンス違反疑義率 (Compliance Violation Rate)
```python
compliance_rate = compliance_violation_count / total_lp_months
```

#### 事務ミス率 (Administrative Error Rate)
```python
jimu_miss_rate = jimu_miss_count / total_lp_months
```

### 3. 機械学習分析 (Machine Learning Analysis)

- **モデル**: XGBoost Classifier (利用可能な場合) または Decision Tree Classifier
- **特徴量**: 4つのリスク指標（連続値）
- **目標変数**: 翌年度懲戒処分フラグ
- **クラス平衡**: SMOTE を使用してクラス不均衡を調整
- **評価指標**: Accuracy, F1-score, Confusion Matrix

### 4. 高度な分析機能

- **SHAP分析**: 特徴量の寄与度と解釈可能性
- **相関分析**: 4つのリスク指標間の相関関係
- **統計的検定**: 各リスク指標と懲戒処分の関連性
- **特徴重要度**: どのリスク指標が最も予測に寄与するか

## ファイル構成 (File Structure)

```
hypothesis_testing/hypothesis_6/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_6.py      # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_6/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_6 \
    --jimu-miss-path /path/to/事務ミスデータ_hashed.xlsx \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_6/run.sh
./hypothesis_testing/hypothesis_6/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_6 \
    --jimu-miss-path /path/to/事務ミスデータ_hashed.xlsx \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, RANK_x, JOB_YYY, JOB_MM, コンプライアンス 列が必要

2. **報酬データ_hashed.csv**
   - SYAIN_CODE, S_YR, S_MO 列が必要

3. **懲戒処分_事故区分等追加_hashed.csv**
   - LP NO, SHOBUN 列が必要

4. **MTG出席率データ** (オプション)
   - LP, S_YR, S_MO, MTG_ATTENDANCE 列が必要
   - ファイル名: MTG出席率2021-2023_hashed.csv など

5. **苦情データ_hashed.xlsx** (オプション)
   - LP, 苦情 列が必要

6. **★事務ミスデータ_不要データ削除版_hashed.xlsx** (重要)
   - LP識別子と日付情報を含む列が必要
   - 存在するレコード = 事務ミスあり、存在しない = 事務ミスなし

### 事務ミスデータの特別な処理

事務ミスデータは存在ベースの論理を使用：
- **存在する場合**: その月に事務ミスが発生
- **存在しない場合**: その月に事務ミスなし（正常）
- データの有無自体が重要な情報

## 出力結果 (Output Results)

### hypothesis_6_results.json

```json
{
  "hypothesis_id": "H6_multidimensional_risk",
  "data_shape": [行数, 列数],
  "performance_metrics": {
    "MTG_Attendance_Rate": {
      "mean": 平均値,
      "median": 中央値,
      "std": 標準偏差
    },
    "苦情_Rate": { ... },
    "コンプライアンス_Rate": { ... },
    "事務ミス_Rate": { ... }
  },
  "model_results": {
    "model_type": "XGBoost",
    "accuracy": 精度,
    "f1_score": "F1スコア",
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "feature_importance": {
      "MTG_Attendance_Rate": 重要度,
      "苦情_Rate": 重要度,
      "コンプライアンス_Rate": 重要度,
      "事務ミス_Rate": 重要度
    }
  },
  "correlation_analysis": {
    "correlation_matrix": 相関行列,
    "high_correlations": 高相関ペア,
    "target_correlations": 目標変数との相関
  },
  "shap_analysis": {
    "shap_importance": SHAP重要度,
    "feature_columns": 特徴量リスト
  },
  "statistical_tests": {
    "MTG_Attendance_Rate": {
      "group_0_mean": 懲戒処分なし群の平均,
      "group_1_mean": 懲戒処分あり群の平均,
      "t_statistic": "t統計量",
      "p_value": "p値",
      "significant": 統計的有意性
    }
  },
  "multidimensional_analysis": {
    "total_amgr_years": 総AMGR年数,
    "amgrs_with_disciplinary_actions": 懲戒処分ありの数,
    "disciplinary_action_rate": 懲戒処分率,
    "prediction_accuracy": 予測精度,
    "f1_score": "F1スコア",
    "feature_count": 特徴量数
  },
  "conclusion": "結論文"
}
```

## 解釈ガイド (Interpretation Guide)

### 成功指標

- **高い予測精度**: Accuracy > 0.7, F1-score > 0.6
- **明確な特徴重要度**: 4つの指標の相対的重要性が明確
- **統計的有意性**: 主要指標のp値 < 0.05
- **相関分析**: 指標間の関係性が理解可能

### 特徴重要度の解釈

1. **最重要指標**: 最も予測に寄与するリスク要因
2. **相補的指標**: 組み合わせで効果を発揮する指標
3. **独立的指標**: 他と相関が低く独自の情報を提供

### ビジネス洞察

1. **予防的管理**: 先行指標による早期警告システム
2. **リソース配分**: 重要度に基づく管理リソースの優先順位
3. **改善施策**: 各指標に対応した具体的な改善アクション

## 技術的詳細 (Technical Details)

### 依存関係

- pandas: データ処理
- numpy: 数値計算
- scikit-learn: 機械学習
- xgboost: 勾配ブースティング (オプション)
- imbalanced-learn: クラス平衡 (オプション)
- shap: 特徴量解釈 (オプション)
- scipy: 統計検定
- openpyxl: Excel読み込み

### アルゴリズム詳細

1. **会計年度計算**: 4月-3月ベースの年度定義
2. **時間オフセット**: 当年→翌年の予測モデル
3. **多次元分析**: 4つの連続値特徴量の組み合わせ効果
4. **SHAP解釈**: 個別予測の説明可能性

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **事務ミスファイルが見つからない**
   ```bash
   # ダミーデータで実行
   python run.py --data-path data --output-path output
   ```

2. **Excel読み込みエラー**
   ```bash
   pip install openpyxl xlrd
   ```

3. **特徴量不足エラー**
   - 4つの指標すべてが計算可能か確認
   - データの欠損状況を確認

### ログの確認

- `--verbose` フラグで詳細ログを出力
- 各段階での処理状況を確認

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - 4次元リスク分析機能
  - XGBoost サポート
  - SHAP 解釈機能
  - 事務ミスデータ統合
