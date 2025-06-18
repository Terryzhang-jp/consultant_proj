#!/usr/bin/env python3
"""
测试Hypothesis 1的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypothesis_testing.hypothesis_1.hypothesis_1 import Hypothesis1ValidatorCorrect
import json

def test_hypothesis_1():
    """测试Hypothesis 1"""
    print("=" * 60)
    print("测试 Hypothesis 1: LP持续低业绩预测懲戒処分")
    print("=" * 60)
    
    try:
        # 创建验证器
        validator = Hypothesis1ValidatorCorrect()
        
        # 运行验证
        results = validator.run_hypothesis_1_with_data_loader()
        
        # 打印结果
        print("\n" + "=" * 40)
        print("验证结果:")
        print("=" * 40)
        
        print(f"假设ID: {results['hypothesis_id']}")
        print(f"数据形状: {results['data_shape']}")
        print(f"验证时间: {results['validated_at']}")
        
        # 打印持续低业绩分析
        sustained_analysis = results['sustained_low_analysis']
        print(f"\n持续低业绩分析:")
        print(f"  总LP数: {sustained_analysis['total_lps']}")
        print(f"  持续低业绩案例数: {sustained_analysis['sustained_low_count']}")
        print(f"  下期懲戒処分案例数: {sustained_analysis['shobun_next_half_count']}")
        print(f"  两者都满足的案例数: {sustained_analysis['both_true_count']}")
        
        # 打印预测准确性
        prediction_accuracy = sustained_analysis['prediction_accuracy']
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
        
        # 打印结论
        print(f"\n结论:")
        print(f"  {results['conclusion']}")
        
        print("\n" + "=" * 60)
        print("Hypothesis 1 测试完成!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hypothesis_1()
    sys.exit(0 if success else 1)
