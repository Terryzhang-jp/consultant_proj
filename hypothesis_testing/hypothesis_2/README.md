# 仮説2: LP収入集中度による懲戒処分予測

## 仮説定義

**仮説**: `if "(変数1) = 1" then "(変数2) >= 1"`

- **変数1 (Variable_1)**: LP(2011-2023データ)のうち、1年のSASHIHIKIの50%以上を2か月で稼いでいる
- **変数2 (Variable_2)**: そのLPが翌年1年間で懲戒処分を受ける回数

---

## コード構造

```
hypothesis_2/
├── __init__.py                  # モジュール初期化
├── hypothesis_validator_2.py    # メイン検証クラス
└── README.md                   # 本ドキュメント
```

### 核心クラスとメソッド

#### `Hypothesis2Validator`
- **メインメソッド**: `validate_hypothesis_2_income_concentration()`
- **核心ロジック**: `_analyze_income_concentration_with_discipline()`
- **統計分析**: `_perform_hypothesis_2_statistical_test()`
- **結果解釈**: `_interpret_h2_results()`

---

## 入力データ仕様

### データ要件
```python
final_dataset = pd.DataFrame({
    'LP': ['LP_0001', 'LP_0002', ...],           # Life Planner ID
    'S_YR': [2011, 2012, 2013, ...],             # 年度 (2011-2023)
    'S_MO': [1, 2, 3, ..., 12],                  # 月 (1-12)
    'SASHIHIKI': [250000, 300000, ...],          # 実支給額
    'SHOBUN': ['警告', '注意', NaN, ...],         # 懲戒処分
})
```

### データフィルタ条件
- **時間範囲**: 2011-2023年データのみ分析
- **データ完整性**: 各LP-年度で最低2か月のSASHIHIKIデータが必要
- **必須カラム**: LP, S_YR, S_MO, SASHIHIKI, SHOBUN

---

## 出力データ仕様

### メイン出力構造
```python
{
    'hypothesis_id': 'H2_income_concentration',
    'data_shape': [495, 8],                     # 処理後データ形状 (LP-年度記録)

    'income_concentration_analysis': {
        'total_lps': 111,                       # 総LP数
        'high_concentration_count': 493,        # Variable_1=1のケース数
        'future_discipline_count': 0,           # Variable_2>=1のケース数
        'both_conditions_met': 0,               # 両条件満足のケース数
        'prediction_accuracy': {
            'accuracy': 0.004,                  # 全体精度
            'hypothesis_accuracy': 0.000,      # 仮説精度
            'precision': 0.000,                 # 適合率
            'recall': 0.000                     # 再現率
        }
    },

    'statistical_test': {
        'test_type': 'Hypothesis 2 Validation: if Variable_1 = 1 then Variable_2 >= 1',
        'confusion_matrix': [[2, 493], [0, 0]], # 混同行列
        'f1_score': 0.000,                      # F1スコア
        'condition_true_cases': 493,            # Variable_1=1のケース数
        'hypothesis_support_rate': 0.000        # 仮説支持率
    },

    'trend_analysis': {
        'yearly_trends': DataFrame              # 年度別トレンド分析
    },

    'conclusion': 'Hypothesis 2 Results: 493 LPs with high income concentration...'
}
```

---

## 処理ロジック

### ステップ1: データ前処理とフィルタ
```python
# 1. 2011-2023年データのフィルタ
df = dataframe[(dataframe['S_YR'] >= 2011) & (dataframe['S_YR'] <= 2023)]

# 2. 必須カラムの確認
needed_cols = ['LP', 'S_YR', 'S_MO', 'SASHIHIKI', 'SHOBUN']
```

### ステップ2: 年度収入分析
```python
# 3. LPと年度でグループ化
for (lp, year), group in df.groupby(['LP', 'S_YR']):

    # 4. 月別SASHIHIKI計算
    monthly_sashihiki = group.groupby('S_MO')['SASHIHIKI'].sum()

    # 5. 年度総収入計算
    annual_total = monthly_sashihiki.sum()
```

### ステップ3: 収入集中度計算
```python
    # 6. 収入最高2か月の特定
    top_2_months = monthly_sashihiki.nlargest(2)
    top_2_total = top_2_months.sum()

    # 7. 集中度比率計算
    concentration_ratio = top_2_total / annual_total

    # 8. 高集中度判定 (50%閾値)
    variable_1 = 1 if concentration_ratio >= 0.5 else 0
```

### ステップ4: 将来懲戒処分チェック
```python
    # 9. 翌年懲戒処分の確認
    next_year = year + 1
    next_year_discipline = df[
        (df['LP'] == lp) &
        (df['S_YR'] == next_year) &
        (df['SHOBUN'].notna())
    ]

    # 10. 翌年懲戒処分回数計算
    variable_2 = len(next_year_discipline)
```

### ステップ5: データ集約と分析
```python
    # 11. LP-年度データ集約
    annual_data.append({
        'LP': lp,
        'Year': year,
        'Annual_SASHIHIKI': annual_total,
        'Top_2_Months_SASHIHIKI': top_2_total,
        'Concentration_Ratio': concentration_ratio,
        'Variable_1': variable_1,           # 高集中度フラグ
        'Variable_2': variable_2,           # 将来懲戒処分回数
        'Top_2_Months': list(top_2_months.index)
    })
```

### ステップ6: 仮説検証
```python
# 12. 仮説検証: Variable_1=1時、Variable_2>=1か
hypothesis_validation = validate(Variable_1, Variable_2)
```

---

## 実装特性

### 収入集中度分析
- **時間窓**: 年度別分析 (2011-2023)
- **集中度指標**: 最高2か月収入が年度総収入に占める比率
- **閾値設定**: 50%を高集中度判定基準とする
- **将来予測**: 翌年懲戒処分状況の確認

### 統計分析機能
- **混同行列**: 予測性能評価
- **F1スコア**: 適合率と再現率のバランス
- **仮説支持率**: 条件満足時の結果発生比率
- **年度トレンド**: 年度別収入集中度と懲戒処分トレンド

### データ処理特徴
- **月度集約**: 月度SASHIHIKIデータの年度分析への集約
- **時間整合**: 収入データと懲戒処分データの時間対応関係確保
- **欠損値処理**: 不完全月度データの適切な処理

---

## 使用例

```python
from hypothesis_testing.hypothesis_2 import Hypothesis2Validator

# 検証器初期化
validator = Hypothesis2Validator()

# 仮説検証実行
results = validator.validate_hypothesis_2_income_concentration(final_dataset)

# 結果確認
print(f"High concentration cases: {results['income_concentration_analysis']['high_concentration_count']}")
print(f"Future discipline cases: {results['income_concentration_analysis']['future_discipline_count']}")
print(f"Hypothesis accuracy: {results['income_concentration_analysis']['prediction_accuracy']['hypothesis_accuracy']}")

# 年度トレンド確認
trend_data = results['trend_analysis']['yearly_trends']
print(trend_data)
```
