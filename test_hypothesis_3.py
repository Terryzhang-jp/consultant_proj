#!/usr/bin/env python3
"""
测试Hypothesis 3的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypothesis_testing.hypothesis_3.hypothesis_3 import Hypothesis3Validator
import json

def test_hypothesis_3():
    """测试Hypothesis 3"""
    print("=" * 60)
    print("测试 Hypothesis 3: AMGR历史懲戒処分影响LP懲戒処分")
    print("=" * 60)
    
    try:
        # 创建验证器
        validator = Hypothesis3Validator()
        
        # 运行验证
        results = validator.run_hypothesis_3_with_data_loader()
        
        # 打印结果
        print("\n" + "=" * 40)
        print("验证结果:")
        print("=" * 40)
        
        print(f"假设ID: {results['hypothesis_id']}")
        print(f"数据形状: {results['data_shape']}")
        print(f"验证时间: {results['validated_at']}")
        
        # 打印AMGR影响分析
        amgr_analysis = results['amgr_influence_analysis']
        print(f"\nAMGR影响分析:")
        print(f"  总AMGR数: {amgr_analysis['total_amgrs']}")
        print(f"  有懲戒処分历史的AMGR数: {amgr_analysis['amgrs_with_history']}")
        print(f"  有LP违规的AMGR数: {amgr_analysis['amgrs_with_lp_violations']}")
        print(f"  两者都满足的AMGR数: {amgr_analysis['both_conditions_met']}")
        
        # 打印预测准确性
        prediction_accuracy = amgr_analysis['prediction_accuracy']
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
        print(f"  两者都满足案例数: {statistical_test['both_true_cases']}")
        print(f"  假设支持率: {statistical_test['hypothesis_support_rate']:.3f}")
        print(f"  混淆矩阵: {statistical_test['confusion_matrix']}")
        
        # 打印AMGR模式分析
        if 'amgr_patterns' in results:
            amgr_patterns = results['amgr_patterns']
            print(f"\nAMGR模式分析:")
            if 'amgr_distribution' in amgr_patterns:
                print(f"  AMGR历史分布: {amgr_patterns['amgr_distribution']}")
            if 'lp_violation_distribution' in amgr_patterns:
                print(f"  LP违规分布: {amgr_patterns['lp_violation_distribution']}")
        
        # 打印性能指标
        if 'performance_metrics' in results:
            performance_metrics = results['performance_metrics']
            print(f"\n性能指标:")
            for metric_name, metric_data in performance_metrics.items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    print(f"  {metric_name}:")
                    print(f"    平均值: {metric_data['mean']:.3f}")
                    print(f"    中位数: {metric_data['median']:.3f}")
                    print(f"    标准差: {metric_data['std']:.3f}")
        
        # 打印结论
        print(f"\n结论:")
        print(f"  {results['conclusion']}")
        
        print("\n" + "=" * 60)
        print("Hypothesis 3 测试完成!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hypothesis_3()
    sys.exit(0 if success else 1)
