# Hypothesis 4: Decision Tree Prediction Analysis

## 概要 (Overview)

Hypothesis 4 は決定木モデルを使用して、LP の入賞率とコンプライアンス違反率から将来の懲戒処分を予測する分析です。

**仮説**: "变数1,2を利用した決定木による判定" then (变数3) = 1

### 変数定義 (Variable Definitions)

- **变数1**: 在籍年数の半分以上で入賞したLPの割合 (LP Award Rate)
  - LP が在籍年数の半分以上で入賞している割合
  - 高い値は優秀な LP の多い組織を示す

- **变数2**: LPひとりあたりの平均年間コンプライアンス違反疑件数 (Compliance Violation Rate)
  - LP 一人当たりの年間平均コンプライアンス違反数
  - 高い値は問題のある組織を示す

- **变数3**: 営業所がその後1年間で受ける懲戒処分の有無 (Future Disciplinary Action)
  - 営業所が翌年に懲戒処分を受けるかどうか
  - 1: 懲戒処分あり, 0: 懲戒処分なし

## 分析ロジック (Analysis Logic)

### 1. データ準備 (Data Preparation)

1. **在職期間と入賞統計の計算**
   - 各 LP の在職年数を計算
   - 入賞年数を計算
   - 入賞年数が在職年数の半分以上かつコンプライアンス違反がない LP を「資格あり」とマーク

2. **AMGR 資格率の計算**
   - 各 AMGR と会計年度について
   - その時点までのデータのみを使用
   - 資格ありの LP の割合を計算

3. **AMGR 違反率の計算**
   - 各 AMGR と会計年度について
   - コンプライアンス違反数を集計
   - LP 一人当たりの違反率を計算

### 2. 予測データの準備 (Prediction Data Preparation)

- 各 AMGR の連続する年度のデータをペアにする
- 現在年度の資格率と違反率を特徴量とする
- 翌年度の懲戒処分有無を目標変数とする

### 3. 決定木モデルの訓練 (Decision Tree Model Training)

- **モデル**: XGBoost Classifier (利用可能な場合) または Decision Tree Classifier
- **特徴量**: 資格率、違反率
- **目標変数**: 翌年の懲戒処分有無
- **評価指標**: 精度、F1スコア、混同行列

## ファイル構成 (File Structure)

```
hypothesis_testing/hypothesis_4/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_4.py      # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_4/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_4 \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_4/run.sh
./hypothesis_testing/hypothesis_4/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_4 \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, RANK, JOB_YYY, JOB_MM, JOB_DD 列が必要

2. **報酬データ_hashed.csv**
   - SYAIN_CODE, S_YR, S_MO, SASHIHIKI, KYUYO 列が必要

3. **懲戒処分_事故区分等追加_hashed.csv**
   - LP NO, FUKABI, SHOBUN 列が必要

4. **社長杯入賞履歴_LPコード0埋_hashed.csv** (オプション)
   - LP, CONTEST_YYY, CONTEST_MM, CONTEST_ID 列が必要

### データ形式要件

- CSV ファイル形式
- 文字エンコーディング: UTF-8
- 日付形式: YYYY/MM/DD または YYYYMM

## 出力結果 (Output Results)

### hypothesis_4_results.json

```json
{
  "hypothesis_id": "H4_decision_tree_prediction",
  "data_shape": [行数, 列数],
  "performance_metrics": {
    "Qualification_Rate": {
      "mean": 平均値,
      "median": 中央値,
      "std": 標準偏差,
      "min": 最小値,
      "max": 最大値
    },
    "Compliance_Violations_Rate": { ... }
  },
  "model_results": {
    "model_type": "XGBoost",
    "accuracy": 精度,
    "f1_score": "F1スコア",
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "feature_importance": {
      "Qualification_Rate": 重要度,
      "Compliance_Violations_Rate": 重要度
    }
  },
  "decision_tree_analysis": {
    "total_records": 総レコード数,
    "prediction_accuracy": 予測精度,
    "f1_score": "F1スコア",
    "feature_count": 特徴量数
  },
  "conclusion": "結論文"
}
```

## 解釈ガイド (Interpretation Guide)

### 成功指標

- **高い精度 (Accuracy > 0.7)**: モデルが将来の懲戒処分を正確に予測
- **高い F1 スコア (F1 > 0.5)**: バランスの取れた予測性能
- **明確な特徴量重要度**: どの変数が予測に重要かが明確

### 特徴量重要度の解釈

- **資格率の重要度が高い**: 優秀な LP の割合が将来の懲戒処分に影響
- **違反率の重要度が高い**: 過去の違反が将来の懲戒処分を予測

### ビジネス洞察

1. **予防的管理**: 高リスクの組織を事前に特定
2. **リソース配分**: 問題のある組織に重点的にサポート
3. **人材育成**: 優秀な LP の育成が組織の健全性に寄与

## 技術的詳細 (Technical Details)

### 依存関係

- pandas: データ処理
- numpy: 数値計算
- scikit-learn: 機械学習
- xgboost: 勾配ブースティング (オプション)
- tqdm: プログレスバー

### パフォーマンス考慮事項

- 大規模データセット対応
- メモリ効率的な処理
- プログレスバーによる進捗表示

### エラーハンドリング

- データ不足時の適切なエラーメッセージ
- 不正なデータ形式の検出
- モデル訓練失敗時の代替処理

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **XGBoost インストールエラー**
   ```bash
   pip install xgboost
   ```

2. **データファイルが見つからない**
   - ファイルパスを確認
   - ファイル名の大文字小文字を確認

3. **メモリ不足エラー**
   - データサイズを削減
   - チャンクサイズを調整

### ログの確認

- `--verbose` フラグを使用して詳細ログを出力
- エラーメッセージを確認してデバッグ

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - 基本的な決定木予測機能
  - XGBoost サポート
  - 詳細な結果出力
