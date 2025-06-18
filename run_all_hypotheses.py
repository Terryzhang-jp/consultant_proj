#!/usr/bin/env python3
"""
运行所有hypothesis的统一测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hypothesis_testing.hypothesis_1.hypothesis_1 import Hypothesis1ValidatorCorrect
from hypothesis_testing.hypothesis_2.hypothesis_2 import Hypothesis2Validator
import json
import traceback

def test_hypothesis_1():
    """测试Hypothesis 1"""
    print("=" * 60)
    print("测试 Hypothesis 1: LP持续低业绩预测懲戒処分")
    print("=" * 60)
    
    try:
        validator = Hypothesis1ValidatorCorrect()
        results = validator.run_hypothesis_1_with_data_loader()
        
        print(f"✅ Hypothesis 1 成功完成")
        print(f"   数据形状: {results['data_shape']}")
        print(f"   持续低业绩案例: {results['sustained_low_analysis']['sustained_low_count']}")
        print(f"   假设准确性: {results['sustained_low_analysis']['prediction_accuracy']['hypothesis_accuracy']:.3f}")
        print(f"   F1分数: {results['statistical_test']['f1_score']:.3f}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Hypothesis 1 失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_hypothesis_2():
    """测试Hypothesis 2"""
    print("\n" + "=" * 60)
    print("测试 Hypothesis 2: LP收入集中度预测懲戒処分")
    print("=" * 60)
    
    try:
        validator = Hypothesis2Validator()
        results = validator.run_hypothesis_2_with_data_loader()
        
        print(f"✅ Hypothesis 2 成功完成")
        print(f"   数据形状: {results['data_shape']}")
        print(f"   高集中度案例: {results['income_concentration_analysis']['high_concentration_count']}")
        print(f"   假设准确性: {results['income_concentration_analysis']['prediction_accuracy']['hypothesis_accuracy']:.3f}")
        print(f"   F1分数: {results['statistical_test']['f1_score']:.3f}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Hypothesis 2 失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_hypothesis_placeholder(hypothesis_num):
    """占位符测试函数"""
    print(f"\n" + "=" * 60)
    print(f"测试 Hypothesis {hypothesis_num}: 尚未实现")
    print("=" * 60)
    print(f"⚠️  Hypothesis {hypothesis_num} 尚未更新为使用统一数据加载器")
    return False, None

def run_all_hypotheses():
    """运行所有hypothesis测试"""
    print("🚀 开始运行所有Hypothesis测试...")
    print("使用统一数据加载器和规范化数据")
    
    results = {}
    success_count = 0
    total_count = 0
    
    # 测试已实现的hypothesis
    hypothesis_tests = [
        ("Hypothesis_1", test_hypothesis_1),
        ("Hypothesis_2", test_hypothesis_2),
        ("Hypothesis_3", lambda: test_hypothesis_placeholder(3)),
        ("Hypothesis_4", lambda: test_hypothesis_placeholder(4)),
        ("Hypothesis_5", lambda: test_hypothesis_placeholder(5)),
        ("Hypothesis_6", lambda: test_hypothesis_placeholder(6)),
        ("Hypothesis_7", lambda: test_hypothesis_placeholder(7)),
        ("Hypothesis_8", lambda: test_hypothesis_placeholder(8)),
        ("Hypothesis_9", lambda: test_hypothesis_placeholder(9)),
        ("Hypothesis_10", lambda: test_hypothesis_placeholder(10)),
    ]
    
    for hypothesis_name, test_func in hypothesis_tests:
        total_count += 1
        success, result = test_func()
        
        results[hypothesis_name] = {
            'success': success,
            'result': result
        }
        
        if success:
            success_count += 1
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print(f"总测试数: {total_count}")
    print(f"成功数: {success_count}")
    print(f"失败数: {total_count - success_count}")
    print(f"成功率: {success_count/total_count*100:.1f}%")
    
    print("\n详细结果:")
    for hypothesis_name, result_info in results.items():
        status = "✅ 成功" if result_info['success'] else "❌ 失败"
        print(f"  {hypothesis_name}: {status}")
    
    # 保存结果到文件
    try:
        # 创建可序列化的结果
        serializable_results = {}
        for hypothesis_name, result_info in results.items():
            serializable_results[hypothesis_name] = {
                'success': result_info['success'],
                'has_result': result_info['result'] is not None
            }
        
        with open('hypothesis_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 结果已保存到: hypothesis_test_results.json")
        
    except Exception as e:
        print(f"⚠️  保存结果文件失败: {str(e)}")
    
    print("\n" + "=" * 80)
    print("🎯 下一步建议:")
    print("=" * 80)
    print("1. 更新其他hypothesis (3-10) 以使用统一数据加载器")
    print("2. 修复任何失败的hypothesis")
    print("3. 优化数据生成以提高hypothesis准确性")
    print("4. 添加更多测试数据验证")
    
    return success_count == total_count

if __name__ == "__main__":
    success = run_all_hypotheses()
    sys.exit(0 if success else 1)
