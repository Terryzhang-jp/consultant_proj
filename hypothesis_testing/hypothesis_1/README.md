# 仮説1: LP継続的低業績による懲戒処分予測

## 仮説定義

**仮説**: `if "Sustained_Low >= 1" then "SHOBUN_in_Next_Half >= 1"`

- **変数1 (Sustained_Low)**: LP(RANK_x==10)のうち、所属するAMGRのSASHIHIKI平均値より低い半年単位のブロックが連続6回以上
- **変数2 (SHOBUN_in_Next_Half)**: 次の半年間で懲戒処分を受ける回数

---

## コード構造

```
hypothesis_1/
├── __init__.py                     # モジュール初期化
├── hypothesis_validator_correct.py # メイン検証クラス
└── README.md                       # 本ドキュメント
```

### 核心クラスとメソッド

#### `Hypothesis1ValidatorCorrect`
- **メインメソッド**: `validate_hypothesis_1_performance_prediction()`
- **核心ロジック**: `_analyze_lp_sashihiki_with_shobun_optimized()`
- **統計分析**: `_perform_hypothesis_1_statistical_test()`
- **結果解釈**: `_interpret_h1_results_correct()`

---

## 入力データ仕様

### データ要件
```python
final_dataset = pd.DataFrame({
    'LP': ['LP_0001', 'LP_0002', ...],           # Life Planner ID
    'RANK_x': [10, 10, 10, ...],                 # 従業員ランク (RANK_x==10のみ処理)
    'AMGR': ['LP_1001', 'LP_1002', ...],         # Area Manager ID
    'SASHIHIKI': [250000, 300000, ...],          # 実支給額
    'S_YR': [2020, 2021, ...],                   # 年度
    'S_MO': [4, 5, 6, ...],                      # 月
    'SHOBUN': ['警告', '注意', NaN, ...],         # 懲戒処分
    'FISCAL_HALF': ['2020_H1', '2020_H2', ...]   # 会計半期
})
```

### データフィルタ条件
- **RANKフィルタ**: `RANK_x == 10` の従業員のみ分析
- **時系列要件**: 連続6半期のデータが必要
- **必須カラム**: LP, RANK_x, AMGR, SASHIHIKI, S_YR, S_MO, SHOBUN

---

## 出力データ仕様

### メイン出力構造
```python
{
    'hypothesis_id': 'H1_performance_prediction',
    'data_shape': [284, 15],                    # 処理後データ形状

    'sustained_low_analysis': {
        'total_lps': 49,                        # 総LP数 (RANK_x==10)
        'sustained_low_count': 76,              # Sustained_Low=1のケース数
        'shobun_next_half_count': 9,            # SHOBUN_in_Next_Half=1のケース数
        'both_true_count': 0,                   # 両条件満足のケース数
        'prediction_accuracy': {
            'accuracy': 0.987,                  # 全体精度
            'hypothesis_accuracy': 0.000,      # 仮説精度
            'precision': 0.000,                 # 適合率
            'recall': 0.000                     # 再現率
        }
    },

    'statistical_test': {
        'test_type': 'Confusion Matrix Analysis (EXACT from mid.py)',
        'confusion_matrix': [[275, 0], [9, 0]], # 混同行列
        'f1_score': 0.000,                      # F1スコア
        'condition_true_cases': 76,             # Sustained_Low=1のケース数
        'hypothesis_support_rate': 0.000        # 仮説支持率
    },

    'conclusion': 'Hypothesis 1 Results: 76 LPs with sustained low performance...'
}
```

---

## 処理ロジック

### ステップ1: データ前処理
```python
# 1. RANK_x==10の従業員をフィルタ
lp_df = dataframe[dataframe['RANK_x'] == 10][needed_cols].copy()

# 2. 会計半期の作成
FISCAL_HALF = if (4 <= S_MO <= 9): S_YR + '_H1'
              elif (S_MO < 4): (S_YR-1) + '_H2'
              else: S_YR + '_H2'
```

### ステップ2: 平均値計算
```python
# 3. LP半期別平均SASHIHIKI計算
lp_avg_sashihiki = groupby(['LP', 'FISCAL_HALF', 'AMGR'])['SASHIHIKI'].mean()

# 4. AMGR半期別平均SASHIHIKI計算
amgr_avg_sashihiki = groupby(['AMGR', 'FISCAL_HALF'])['SASHIHIKI'].mean()
```

### ステップ3: 比較とローリング窓分析
```python
# 5. AMGR平均値以下の判定
Below_AMGR_Avg = (lp_sashihiki < amgr_avg_sashihiki)

# 6. 連続低業績半期数の計算 (重要: min_periods=6)
Rolling_Low = rolling(window=6, min_periods=6).sum()

# 7. Sustained_Low: 連続6半期低業績
Sustained_Low = (Rolling_Low >= 6)
```

### ステップ4: 将来懲戒処分チェック
```python
# 8. 次半期懲戒処分の確認
next_half_mapping = {'2020_H1': '2020_H2', '2020_H2': '2021_H1', ...}

# 9. SHOBUN_in_Next_Half: 次期懲戒処分有無
SHOBUN_in_Next_Half = has_shobun_in_next_half
```

### ステップ5: 仮説検証
```python
# 10. 仮説検証: Sustained_Low=1時、SHOBUN_in_Next_Half=1か
hypothesis_validation = validate(Sustained_Low, SHOBUN_in_Next_Half)
```

---

## 実装特性

### 中継コードとの完全一致
- ✅ **比較基準**: AMGR (OFFICEではない)
- ✅ **データフィルタ**: RANK_x == 10のみ分析
- ✅ **ローリング窓**: min_periods=6 (6完全データポイント必須)
- ✅ **変数名**: Sustained_Low, SHOBUN_in_Next_Half
- ✅ **全計算ロジック**: 原始コードと完全一致

### 統計分析機能
- **混同行列**: 予測性能評価
- **F1スコア**: 適合率と再現率のバランス
- **仮説支持率**: 条件満足時の結果発生比率
- **トレンド分析**: 会計半期別トレンド変化

---

## 使用例

```python
from hypothesis_testing.hypothesis_1 import Hypothesis1ValidatorCorrect

# 検証器初期化
validator = Hypothesis1ValidatorCorrect()

# 仮説検証実行
results = validator.validate_hypothesis_1_performance_prediction(final_dataset)

# 結果確認
print(f"Sustained Low cases: {results['sustained_low_analysis']['sustained_low_count']}")
print(f"Hypothesis accuracy: {results['sustained_low_analysis']['prediction_accuracy']['hypothesis_accuracy']}")
```
