#!/usr/bin/env python3
"""
测试Hypothesis 2的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypothesis_testing.hypothesis_2.hypothesis_2 import Hypothesis2Validator
import json

def test_hypothesis_2():
    """测试Hypothesis 2"""
    print("=" * 60)
    print("测试 Hypothesis 2: LP收入集中度预测懲戒処分")
    print("=" * 60)
    
    try:
        # 创建验证器
        validator = Hypothesis2Validator()
        
        # 运行验证
        results = validator.run_hypothesis_2_with_data_loader()
        
        # 打印结果
        print("\n" + "=" * 40)
        print("验证结果:")
        print("=" * 40)
        
        print(f"假设ID: {results['hypothesis_id']}")
        print(f"数据形状: {results['data_shape']}")
        print(f"验证时间: {results['validated_at']}")
        
        # 打印收入集中度分析
        concentration_analysis = results['income_concentration_analysis']
        print(f"\n收入集中度分析:")
        print(f"  总LP数: {concentration_analysis['total_lps']}")
        print(f"  高集中度案例数: {concentration_analysis['high_concentration_count']}")
        print(f"  未来懲戒処分案例数: {concentration_analysis['future_discipline_count']}")
        print(f"  两者都满足的案例数: {concentration_analysis['both_conditions_met']}")
        
        # 打印预测准确性
        prediction_accuracy = concentration_analysis['prediction_accuracy']
        print(f"\n预测准确性:")
        print(f"  整体准确性: {prediction_accuracy['accuracy']:.3f}")
        print(f"  假设准确性: {prediction_accuracy['hypothesis_accuracy']:.3f}")
        print(f"  精确度: {prediction_accuracy['precision']:.3f}")
        print(f"  召回率: {prediction_accuracy['recall']:.3f}")
        
        # 打印统计测试结果
        statistical_test = results['statistical_test']
        print(f"\n统计测试:")
        print(f"  测试类型: {statistical_test['test_type']}")
        print(f"  F1分数: {statistical_test['f1_score']:.3f}")
        print(f"  条件满足案例数: {statistical_test['condition_true_cases']}")
        print(f"  假设支持率: {statistical_test['hypothesis_support_rate']:.3f}")
        print(f"  混淆矩阵: {statistical_test['confusion_matrix']}")
        
        # 打印年度趋势
        if 'trend_analysis' in results and 'yearly_trends' in results['trend_analysis']:
            trend_data = results['trend_analysis']['yearly_trends']
            print(f"\n年度趋势 (前5年):")
            print(trend_data.head())
        
        # 打印结论
        print(f"\n结论:")
        print(f"  {results['conclusion']}")
        
        print("\n" + "=" * 60)
        print("Hypothesis 2 测试完成!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hypothesis_2()
    sys.exit(0 if success else 1)
