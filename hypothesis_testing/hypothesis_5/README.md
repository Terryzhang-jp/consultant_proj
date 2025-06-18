# Hypothesis 5: AMGR Experience Impact Analysis

## 概要 (Overview)

Hypothesis 5 は AMGR の LP 時代の経験年数が、その後の部下の懲戒処分に与える影響を分析します。

**仮説**: if "(变数1<2)" & then (变数2) =1

### 変数定義 (Variable Definitions)

- **变数1**: AMGRがLPの時代の経験年数 (AMGR's Experience Years as LP)
  - AMGR が LP として働いた年数
  - 2年未満の経験が重要な閾値

- **变数2**: AMGR時代の配下のLPの懲戒処分の有無 (Disciplinary Action of Subordinate LPs under AMGR)
  - AMGR の管理下にある LP の懲戒処分の有無
  - 1: 懲戒処分あり, 0: 懲戒処分なし

## 分析ロジック (Analysis Logic)

### 1. データ準備 (Data Preparation)

1. **LP 経験年数の計算**
   - 各 AMGR について、LP (RANK=10) から AMGR (RANK=20) への昇進履歴を追跡
   - LP として働いた月数を年数に変換

2. **AMGR 管理データの集計**
   - 各 AMGR の管理期間（任期）を計算
   - 管理下の唯一 LP 数を計算
   - 部下の懲戒処分件数を集計

3. **仮説検証データの作成**
   - 経験年数 < 2年 vs ≥ 2年 のカテゴリ分類
   - 懲戒処分フラグの作成

### 2. 機械学習分析 (Machine Learning Analysis)

- **モデル**: XGBoost Classifier (利用可能な場合) または Decision Tree Classifier
- **特徴量**: LP経験年数、AMGR任期年数
- **目標変数**: 懲戒処分フラグ
- **クラス平衡**: SMOTE を使用してクラス不均衡を調整

### 3. 統計的検証 (Statistical Validation)

- **カイ二乗検定**: 経験年数カテゴリと懲戒処分の関連性
- **比率比較**: <2年 vs ≥2年 の懲戒処分率
- **仮説支持度**: 統計的有意性の評価

## ファイル構成 (File Structure)

```
hypothesis_testing/hypothesis_5/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_5.py      # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_5/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_5 \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_5/run.sh
./hypothesis_testing/hypothesis_5/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_5 \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, RANK_x, JOB_YYY, JOB_MM, JOB_DD 列が必要
   - RANK_x: 10=LP, 20=AMGR の昇進履歴

2. **報酬データ_hashed.csv**
   - SYAIN_CODE, S_YR, S_MO, SASHIHIKI, KYUYO 列が必要

3. **懲戒処分_事故区分等追加_hashed.csv**
   - LP NO, FUKABI, SHOBUN 列が必要

4. **社長杯入賞履歴_LPコード0埋_hashed.csv** (オプション)
   - LP, CONTEST_YYY, CONTEST_MM, CONTEST_ID 列が必要

5. **業績_hashed.csv** (オプション)
   - LP, ym, SEISEKI 列が必要

### データ形式要件

- CSV ファイル形式
- 文字エンコーディング: UTF-8
- 日付形式: YYYY/MM/DD または YYYYMM

## 出力結果 (Output Results)

### hypothesis_5_results.json

```json
{
  "hypothesis_id": "H5_amgr_experience_impact",
  "data_shape": [行数, 列数],
  "performance_metrics": {
    "LP_Experience_Before_AMGR": {
      "mean": 平均値,
      "median": 中央値,
      "std": 標準偏差,
      "min": 最小値,
      "max": 最大値
    }
  },
  "model_results": {
    "model_type": "XGBoost",
    "accuracy": 精度,
    "f1_score": "F1スコア",
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "feature_importance": {
      "LP_Experience_Before_AMGR": 重要度,
      "AMGR_Tenure_Years": 重要度
    }
  },
  "hypothesis_test_results": {
    "less_than_2_years_count": 2年未満のAMGR数,
    "two_years_or_more_count": 2年以上のAMGR数,
    "less_than_2_years_disciplinary_rate": 2年未満の懲戒処分率,
    "two_years_or_more_disciplinary_rate": 2年以上の懲戒処分率,
    "hypothesis_supported": 仮説支持の有無,
    "rate_difference": 懲戒処分率の差,
    "chi2_statistic": カイ二乗統計量,
    "p_value": "p値"
  },
  "amgr_experience_analysis": {
    "total_amgrs": 総AMGR数,
    "amgrs_with_less_than_2_years": 2年未満経験のAMGR数,
    "amgrs_with_disciplinary_actions": 懲戒処分ありのAMGR数,
    "prediction_accuracy": 予測精度,
    "f1_score": "F1スコア"
  },
  "conclusion": "結論文"
}
```

## 解釈ガイド (Interpretation Guide)

### 成功指標

- **仮説支持**: `hypothesis_supported: true` かつ統計的有意性
- **高い予測精度**: Accuracy > 0.7, F1-score > 0.5
- **明確な率差**: 2年未満と2年以上の懲戒処分率に有意差

### 統計的解釈

- **p値 < 0.05**: 統計的に有意な関連性
- **率差 > 0**: 2年未満の AMGR の方が懲戒処分率が高い
- **カイ二乗統計量**: 関連性の強さを示す

### ビジネス洞察

1. **採用・昇進基準**: LP 経験年数の重要性
2. **研修プログラム**: 経験不足 AMGR への追加サポート
3. **リスク管理**: 高リスク AMGR の早期識別

## 技術的詳細 (Technical Details)

### 依存関係

- pandas: データ処理
- numpy: 数値計算
- scikit-learn: 機械学習
- xgboost: 勾配ブースティング (オプション)
- imbalanced-learn: クラス平衡 (オプション)
- scipy: 統計検定
- tqdm: プログレスバー

### アルゴリズム詳細

1. **経験年数計算**: 月次データから年数への変換
2. **クラス平衡**: SMOTE による少数クラスの増強
3. **特徴重要度**: XGBoost の内蔵機能を使用
4. **統計検定**: カイ二乗検定による独立性検定

### パフォーマンス考慮事項

- 大規模データセット対応
- メモリ効率的な処理
- 欠損値の適切な処理

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **依存関係エラー**
   ```bash
   pip install xgboost imbalanced-learn scipy
   ```

2. **データ不足エラー**
   - AMGR の昇進履歴データを確認
   - RANK_x 列の値を確認 (10=LP, 20=AMGR)

3. **統計検定エラー**
   - サンプルサイズが小さすぎる場合
   - カテゴリの偏りが極端な場合

### ログの確認

- `--verbose` フラグを使用して詳細ログを出力
- エラーメッセージを確認してデバッグ

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - 基本的な経験年数分析機能
  - XGBoost サポート
  - 統計的検証機能
  - 詳細な結果出力
