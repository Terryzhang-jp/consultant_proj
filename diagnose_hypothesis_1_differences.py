#!/usr/bin/env python3
"""
诊断Hypothesis 1结果差异的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypothesis_testing.hypothesis_1.hypothesis_1 import Hypothesis1ValidatorCorrect
import pandas as pd

def analyze_data_characteristics():
    """分析数据特征"""
    print("🔍 分析数据特征...")
    
    validator = Hypothesis1ValidatorCorrect()
    
    # 加载原始数据
    final_dataset = validator.data_loader.create_final_dataset(
        include_lp_history=True,
        include_reward=True,
        include_discipline=True
    )
    
    print(f"📊 原始数据统计:")
    print(f"   总记录数: {len(final_dataset):,}")
    print(f"   总LP数: {final_dataset['LP'].nunique():,}")
    print(f"   年份范围: {final_dataset['S_YR'].min()}-{final_dataset['S_YR'].max()}")
    print(f"   RANK分布: {final_dataset['RANK_x'].value_counts().to_dict()}")
    
    # 分析RANK=10的数据
    rank_10_data = final_dataset[final_dataset['RANK_x'] == 10]
    print(f"\n📈 RANK=10数据统计:")
    print(f"   RANK=10记录数: {len(rank_10_data):,}")
    print(f"   RANK=10 LP数: {rank_10_data['LP'].nunique():,}")
    print(f"   年份范围: {rank_10_data['S_YR'].min()}-{rank_10_data['S_YR'].max()}")
    
    # 分析懲戒処分数据
    discipline_data = final_dataset[final_dataset['SHOBUN_FLAG'] == 1]
    print(f"\n⚖️  懲戒処分数据统计:")
    print(f"   懲戒処分记录数: {len(discipline_data):,}")
    print(f"   懲戒処分LP数: {discipline_data['LP'].nunique():,}")
    print(f"   年份分布: {discipline_data['S_YR'].value_counts().sort_index().head(10).to_dict()}")
    
    return final_dataset

def analyze_hypothesis_1_processing(final_dataset):
    """分析Hypothesis 1的处理过程"""
    print("\n🎯 分析Hypothesis 1处理过程...")
    
    validator = Hypothesis1ValidatorCorrect()
    
    # 运行分析
    analysis_results = validator._analyze_lp_sashihiki_with_shobun_optimized(final_dataset)
    
    print(f"📋 处理后数据统计:")
    print(f"   分析记录数: {len(analysis_results):,}")
    print(f"   唯一LP数: {analysis_results['LP'].nunique():,}")
    print(f"   唯一FISCAL_HALF数: {analysis_results['FISCAL_HALF'].nunique():,}")
    
    # 分析变数1和变数2
    var1_count = (analysis_results['Sustained_Low'] == 1).sum()
    var2_count = (analysis_results['SHOBUN_in_Next_Half'] == 1).sum()
    both_count = ((analysis_results['Sustained_Low'] == 1) & 
                  (analysis_results['SHOBUN_in_Next_Half'] == 1)).sum()
    
    print(f"\n📊 变数统计:")
    print(f"   变数1 (Sustained_Low=1): {var1_count:,}")
    print(f"   变数2 (SHOBUN_in_Next_Half=1): {var2_count:,}")
    print(f"   两者都满足: {both_count:,}")
    
    # 分析FISCAL_HALF分布
    fiscal_half_dist = analysis_results['FISCAL_HALF'].value_counts().sort_index()
    print(f"\n📅 FISCAL_HALF分布 (前10个):")
    for fiscal_half, count in fiscal_half_dist.head(10).items():
        print(f"   {fiscal_half}: {count:,}")
    
    # 分析年份分布
    year_dist = analysis_results.groupby('FISCAL_HALF').size().reset_index(name='count')
    year_dist['year'] = year_dist['FISCAL_HALF'].str.split('_').str[0].astype(int)
    yearly_summary = year_dist.groupby('year')['count'].sum().sort_index()
    
    print(f"\n📈 年度数据分布 (前10年):")
    for year, count in yearly_summary.head(10).items():
        print(f"   {year}: {count:,}")
    
    return analysis_results

def compare_with_expected_results():
    """与预期结果进行对比"""
    print("\n🔍 与图片结果对比分析...")
    
    # 图片中的预期结果
    expected_results = {
        'total_cases': 110391,
        'discipline_cases': 1448,
        'f1_score': 0.017,
        'total_lps': 10346  # 从描述推断
    }
    
    # 当前结果
    validator = Hypothesis1ValidatorCorrect()
    results = validator.run_hypothesis_1_with_data_loader()
    
    current_results = {
        'total_cases': results['data_shape'][0],
        'discipline_cases': results['sustained_low_analysis']['future_discipline_count'],
        'f1_score': results['statistical_test']['f1_score'],
        'total_lps': results['sustained_low_analysis']['total_lps']
    }
    
    print(f"📊 结果对比:")
    print(f"   指标                  | 图片结果    | 当前结果    | 差异")
    print(f"   --------------------|-----------|-----------|----------")
    print(f"   总案例数             | {expected_results['total_cases']:,}      | {current_results['total_cases']:,}     | {current_results['total_cases'] - expected_results['total_cases']:+,}")
    print(f"   懲戒処分案例数        | {expected_results['discipline_cases']:,}       | {current_results['discipline_cases']:,}      | {current_results['discipline_cases'] - expected_results['discipline_cases']:+,}")
    print(f"   F1分数              | {expected_results['f1_score']:.3f}     | {current_results['f1_score']:.3f}     | {current_results['f1_score'] - expected_results['f1_score']:+.3f}")
    print(f"   总LP数              | {expected_results['total_lps']:,}      | {current_results['total_lps']:,}     | {current_results['total_lps'] - expected_results['total_lps']:+,}")
    
    # 分析可能的原因
    print(f"\n🤔 可能的差异原因:")
    
    if current_results['total_lps'] > expected_results['total_lps']:
        print(f"   1. 当前数据包含更多LP ({current_results['total_lps']:,} vs {expected_results['total_lps']:,})")
    
    if current_results['total_cases'] > expected_results['total_cases']:
        print(f"   2. 当前数据覆盖更长时间范围或更多记录")
    
    if current_results['discipline_cases'] > expected_results['discipline_cases']:
        print(f"   3. 懲戒処分数据匹配方式可能不同")
    
    print(f"   4. 数据过滤条件可能不同（年份范围、RANK过滤等）")
    print(f"   5. 真实数据 vs 测试数据的差异")

def suggest_adjustments():
    """建议调整方案"""
    print(f"\n💡 建议调整方案:")
    print(f"   1. 检查年份过滤范围 - 可能需要限制到特定年份")
    print(f"   2. 检查LP过滤条件 - 可能需要额外的过滤条件")
    print(f"   3. 检查懲戒処分数据匹配逻辑")
    print(f"   4. 验证FISCAL_HALF计算逻辑")
    print(f"   5. 确认rolling window参数（当前是6个半年度）")

def main():
    """主函数"""
    print("🚀 开始诊断Hypothesis 1结果差异...")
    print("=" * 80)
    
    try:
        # 1. 分析数据特征
        final_dataset = analyze_data_characteristics()
        
        # 2. 分析处理过程
        analysis_results = analyze_hypothesis_1_processing(final_dataset)
        
        # 3. 与预期结果对比
        compare_with_expected_results()
        
        # 4. 建议调整方案
        suggest_adjustments()
        
        print("\n" + "=" * 80)
        print("🎯 诊断完成！")
        
    except Exception as e:
        print(f"❌ 诊断过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
