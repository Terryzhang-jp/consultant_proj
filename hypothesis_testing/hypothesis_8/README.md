# Hypothesis 8: Career vs TAP LP Meeting Attendance Analysis

## 概要 (Overview)

Hypothesis 8 はCareer LP和TAP LP的会议出席率差异是否能预测下一年度的纪律处分。

**仮説**: if "(变数3) < 0" then (变数4) = 1

### 変数定義 (Variable Definitions)

- **变数1(t)**: "キャリアLPのMTG出席率" (Career LP Meeting Attendance Rate)
  - 工作年限>3年的LP的会议出席率（连续值）
  - 经验丰富的资深员工群体

- **变数2(t)**: "TAPLPのMTG出席率" (TAP LP Meeting Attendance Rate)
  - 工作年限≤3年的LP的会议出席率（连续值）
  - 新入职或经验较少的员工群体

- **变数3(t)**: 变数1(t) - 变数2(t) (Attendance Difference)
  - Career LP出席率 - TAP LP出席率（连续值）
  - 负值表示TAP LP出席率更高

- **变数4(t)**: 営業所の翌年度の懲戒処分の有無 (Disciplinary Actions in Following Year)
  - 支社下一年度是否发生纪律处分（二元值）
  - 1: 有纪律处分, 0: 无纪律处分

### 重要注意事项

**AMGRとS_YRでグループ化した際、一部のグループにはCareer LPのみ、またはTAP LPのみが存在するため、これらは考慮対象外とする**

- 只有Career LP的组：无法计算差异，排除
- 只有TAP LP的组：无法计算差异，排除
- 同时有Career和TAP LP的组：可以计算差异，纳入分析

## 分析ロジック (Analysis Logic)

### 1. 数据准备流程

1. **工作年限计算**: 基于最早S_YR计算JOIN_YEAR，然后计算Work_Tenure
2. **Career/TAP分类**: Work_Tenure > 3年为Career，≤3年为TAP
3. **支社分组**: 以AMGR为单位定义各支社
4. **出席率计算**: 分别计算Career LP和TAP LP的会议出席率
5. **差值计算**: 计算出席率差异 (Career - TAP)
6. **数据过滤**: 排除只有单一类型LP的组（dropna处理）
7. **时间偏移**: 将纪律处分数据向前偏移一年实现"翌年度"预测

### 2. 核心假设逻辑

```
if (Career LP出席率 - TAP LP出席率) < 0:
    then 翌年度有纪律处分 = 1
```

**业务解释**:
- 当TAP LP比Career LP更积极参与会议时
- 可能反映组织内部的经验传承问题
- 资深员工参与度低可能导致管理问题
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
hypothesis_testing/hypothesis_8/
├── __init__.py                    # モジュール初期化
├── hypothesis_validator_8.py      # メイン検証ロジック
├── run.py                        # Python 実行スクリプト
├── run.sh                        # Shell 実行スクリプト
└── README.md                     # このファイル
```

## 使用方法 (Usage)

### Python スクリプトで実行

```bash
python hypothesis_testing/hypothesis_8/run.py \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_8 \
    --verbose
```

### Shell スクリプトで実行

```bash
chmod +x hypothesis_testing/hypothesis_8/run.sh
./hypothesis_testing/hypothesis_8/run.sh \
    --data-path test_data_folder \
    --output-path hypothesis_testing/hypothesis_8 \
    --verbose
```

## 入力データ要件 (Input Data Requirements)

### 必要なファイル

1. **LPヒストリー_hashed.csv**
   - LP, AMGR, RANK_x, JOB_YYY, JOB_MM, コンプライアンス 列が必要

2. **報酬データ_hashed.csv**
   - SYAIN_CODE, S_YR, S_MO 列が必要
   - 工作年限計算に使用

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
- **時系列データ**: 工作年限計算に必須（最早S_YR）
- **会議出席データ**: 核心分析対象
- **複数年度データ**: 時間偏移のため必要

## 出力結果 (Output Results)

### hypothesis_8_results.json

```json
{
  "hypothesis_id": "H8_career_tap_difference",
  "data_shape": [行数, 列数],
  "descriptive_statistics": {
    "attendance_difference": {
      "mean": 平均差値,
      "median": 中央差値,
      "negative_count": 負値件数,
      "negative_rate": 負値率
    },
    "career_attendance": {
      "mean": "Career LP平均出席率",
      "std": 標準偏差
    },
    "tap_attendance": {
      "mean": "TAP LP平均出席率",
      "std": 標準偏差
    },
    "disciplinary_actions": {
      "total_records": 総記録数,
      "with_disciplinary_actions": 処分ありの数,
      "disciplinary_action_rate": 処分率
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
      "attendance_difference_vs_disciplinary": 相関係数
    }
  },
  "career_tap_analysis": {
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

1. **経験伝承指標**: Career/TAP出席率差異が知識伝承の健全性を反映
2. **世代間バランス**: 資深員工と新人の参与積極性のバランス
3. **早期警告システム**: 負の差異による翌年リスク予測
4. **人材管理**: 経験者の参与促進と新人の指導体制

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

1. **工作年限計算**: groupby('LP')['S_YR'].min() による入社年計算
2. **Career/TAP分類**: Work_Tenure > 3年の閾値判定
3. **データフィルタリング**: dropna()による不完全グループ除外
4. **時間偏移**: groupby('AMGR').shift(-1) による翌年予測
5. **二重検証**: XGBoost + 簡単分類の組み合わせ

## トラブルシューティング (Troubleshooting)

### よくある問題

1. **出席データなし**
   ```bash
   Warning: No attendance column found, using dummy data
   ```

2. **AMGR情報不足**
   ```bash
   ValueError: No data with AMGR information found
   ```

3. **不完全グループ多数**
   - Career/TAP両方存在するグループが少ない
   - データ期間を拡大して解決

### データ品質チェック

- AMGR列の完整性確認
- 時系列データの連続性確認
- Career/TAP分布の妥当性確認
- 出席データの存在確認

## 更新履歴 (Change Log)

- **v1.0.0**: 初期実装
  - Career/TAP分類機能
  - 二重検証モード実装
  - 工作年限ベース分析
  - 不完全グループ除外機能
