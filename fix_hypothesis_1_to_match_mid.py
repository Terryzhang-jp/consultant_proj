#!/usr/bin/env python3
"""
修复Hypothesis 1以匹配mid.py的确切设置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypothesis_testing.hypothesis_1.hypothesis_1 import Hypothesis1ValidatorCorrect
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def create_mid_py_compatible_dataset():
    """创建与mid.py兼容的数据集"""
    print("🔄 创建与mid.py兼容的数据集...")
    
    validator = Hypothesis1ValidatorCorrect()
    
    # 加载完整数据集
    final_dataset = validator.data_loader.create_final_dataset(
        include_lp_history=True,
        include_reward=True,
        include_discipline=True
    )
    
    print(f"📊 原始数据: {len(final_dataset):,} 条记录")
    
    # 关键修复1: 年份过滤 - 只保留2011年及以后的数据（与mid.py第200行一致）
    filtered_dataset = final_dataset[final_dataset['S_YR'] >= 2011].copy()
    print(f"📅 2011年后数据: {len(filtered_dataset):,} 条记录")
    
    # 关键修复2: 列名统一 - 将RANK_x改为RANK（与mid.py一致）
    if 'RANK_x' in filtered_dataset.columns:
        filtered_dataset = filtered_dataset.rename(columns={'RANK_x': 'RANK'})
        print("✅ 列名修复: RANK_x -> RANK")
    
    return filtered_dataset

def analyze_with_mid_py_logic(dataframe):
    """使用与mid.py完全一致的逻辑进行分析"""
    print("🎯 使用mid.py逻辑进行分析...")
    
    # Step 1: 创建FISCAL_HALF（与mid.py完全一致）
    def get_fiscal_half(row):
        if 4 <= row['S_MO'] <= 9:
            return f"{row['S_YR']}_H1"  # Fiscal first half
        else:
            # Fiscal second half
            return f"{row['S_YR'] - 1}_H2" if row['S_MO'] < 4 else f"{row['S_YR']}_H2"
    
    dataframe['FISCAL_HALF'] = dataframe.apply(get_fiscal_half, axis=1)
    print(f"✅ FISCAL_HALF创建完成")
    
    # Step 2: 过滤RANK=10的数据
    needed_cols = ['LP', 'RANK', 'S_YR', 'S_MO', 'SASHIHIKI', 'AMGR', 'SHOBUN', 'FISCAL_HALF']
    lp_df = dataframe[dataframe['RANK'] == 10][needed_cols].copy()
    print(f"📋 RANK=10数据: {len(lp_df):,} 条记录, {lp_df['LP'].nunique():,} 个LP")
    
    # Step 3 & 4: 计算平均值（与mid.py完全一致）
    lp_avg_sashihiki = lp_df.groupby(['LP', 'FISCAL_HALF', 'AMGR'])['SASHIHIKI'].mean().reset_index()
    amgr_avg_sashihiki = dataframe.groupby(['AMGR', 'FISCAL_HALF'])['SASHIHIKI'].mean().reset_index()
    amgr_avg_sashihiki.columns = ['AMGR', 'FISCAL_HALF', 'AMGR_AVG_SASHIHIKI']
    
    # Step 5: 合并数据
    result_df = pd.merge(lp_avg_sashihiki, amgr_avg_sashihiki, on=['AMGR', 'FISCAL_HALF'], how='left')
    
    # Step 6: 比较操作
    result_df['Below_AMGR_Avg'] = result_df['SASHIHIKI'] < result_df['AMGR_AVG_SASHIHIKI']
    
    # Step 7: Rolling window计算（6个连续半年度）
    result_df = result_df.sort_values(['LP', 'FISCAL_HALF'])
    result_df['Rolling_Low'] = result_df.groupby('LP')['Below_AMGR_Avg'].rolling(
        window=6, min_periods=6).sum().reset_index(0, drop=True)
    result_df['Sustained_Low'] = (result_df['Rolling_Low'] >= 6).astype(int)
    
    # Step 8: 检查下一个半年度的SHOBUN
    shobun_df = lp_df[['LP', 'FISCAL_HALF', 'SHOBUN']].dropna(subset=['SHOBUN'])
    shobun_df['Has_SHOBUN'] = True
    
    # 创建下一个半年度的映射
    fiscal_half_order = sorted(result_df['FISCAL_HALF'].unique(), 
                             key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
    next_half_mapping = dict(zip(fiscal_half_order[:-1], fiscal_half_order[1:]))
    result_df['Next_FISCAL_HALF'] = result_df['FISCAL_HALF'].map(next_half_mapping)
    
    # 合并SHOBUN信息
    result_df = pd.merge(
        result_df,
        shobun_df[['LP', 'FISCAL_HALF', 'Has_SHOBUN']],
        left_on=['LP', 'Next_FISCAL_HALF'],
        right_on=['LP', 'FISCAL_HALF'],
        how='left',
        suffixes=('', '_next')
    )
    
    result_df['SHOBUN_in_Next_Half'] = result_df['Has_SHOBUN'].fillna(False).astype(int)
    
    # 清理临时列
    result_df = result_df.drop(['FISCAL_HALF_next', 'Has_SHOBUN', 'Next_FISCAL_HALF', 'Rolling_Low'], axis=1)
    
    print(f"📊 分析完成:")
    print(f"   总记录数: {len(result_df):,}")
    print(f"   唯一LP数: {result_df['LP'].nunique():,}")
    print(f"   Sustained_Low=1: {(result_df['Sustained_Low'] == 1).sum():,}")
    print(f"   SHOBUN_in_Next_Half=1: {(result_df['SHOBUN_in_Next_Half'] == 1).sum():,}")
    
    return result_df

def create_confusion_matrix_analysis(result_df):
    """创建混淆矩阵分析（与mid.py完全一致）"""
    print("\n📈 创建混淆矩阵分析...")
    
    # 创建混淆矩阵（与mid.py完全一致）
    cm = confusion_matrix(result_df['SHOBUN_in_Next_Half'], result_df['Sustained_Low'])
    f1 = f1_score(result_df['SHOBUN_in_Next_Half'], result_df['Sustained_Low'])
    
    print(f"🎯 混淆矩阵:")
    print(cm)
    print(f"📊 F1分数: {f1:.3f}")
    
    # 详细统计
    sustained_low_count = (result_df['Sustained_Low'] == 1).sum()
    shobun_count = (result_df['SHOBUN_in_Next_Half'] == 1).sum()
    both_count = ((result_df['Sustained_Low'] == 1) & (result_df['SHOBUN_in_Next_Half'] == 1)).sum()
    
    print(f"\n📋 详细统计:")
    print(f"   持续低业绩案例数: {sustained_low_count:,}")
    print(f"   下期懲戒処分案例数: {shobun_count:,}")
    print(f"   两者都满足: {both_count:,}")
    print(f"   假设支持率: {both_count/sustained_low_count:.3f}" if sustained_low_count > 0 else "   假设支持率: 0.000")
    
    return {
        'confusion_matrix': cm,
        'f1_score': f1,
        'sustained_low_count': sustained_low_count,
        'shobun_count': shobun_count,
        'both_count': both_count,
        'total_records': len(result_df)
    }

def compare_with_expected():
    """与图片中的预期结果对比"""
    print("\n🔍 与预期结果对比:")
    
    expected = {
        'total_cases': 110391,
        'discipline_cases': 1448,
        'f1_score': 0.017
    }
    
    print(f"   指标                | 预期结果    | 当前结果    | 状态")
    print(f"   -------------------|-----------|-----------|----------")
    
    return expected

def main():
    """主函数"""
    print("🚀 修复Hypothesis 1以匹配mid.py设置...")
    print("=" * 80)
    
    try:
        # 1. 创建兼容数据集
        dataset = create_mid_py_compatible_dataset()
        
        # 2. 使用mid.py逻辑分析
        analysis_results = analyze_with_mid_py_logic(dataset)
        
        # 3. 创建混淆矩阵分析
        stats = create_confusion_matrix_analysis(analysis_results)
        
        # 4. 与预期结果对比
        expected = compare_with_expected()
        
        print(f"\n📊 最终对比:")
        print(f"   总案例数: {stats['total_records']:,} (预期: {expected['total_cases']:,})")
        print(f"   懲戒処分: {stats['shobun_count']:,} (预期: {expected['discipline_cases']:,})")
        print(f"   F1分数: {stats['f1_score']:.3f} (预期: {expected['f1_score']:.3f})")
        
        # 计算差异
        case_diff = stats['total_records'] - expected['total_cases']
        discipline_diff = stats['shobun_count'] - expected['discipline_cases']
        f1_diff = stats['f1_score'] - expected['f1_score']
        
        print(f"\n📈 差异分析:")
        print(f"   总案例差异: {case_diff:+,}")
        print(f"   懲戒処分差异: {discipline_diff:+,}")
        print(f"   F1分数差异: {f1_diff:+.3f}")
        
        if abs(case_diff) < 10000 and abs(discipline_diff) < 500:
            print("\n✅ 结果已接近预期！")
        else:
            print("\n⚠️  仍有差异，可能需要进一步调整数据过滤条件")
        
        print("\n" + "=" * 80)
        print("🎯 修复完成！")
        
    except Exception as e:
        print(f"❌ 修复过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
