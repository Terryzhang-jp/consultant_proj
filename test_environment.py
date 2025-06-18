#!/usr/bin/env python3
"""
环境测试脚本
用于验证项目环境是否正确配置
"""

import sys
import os

def test_imports():
    """测试必要的包导入"""
    print("🔍 测试包导入...")
    
    try:
        import pandas as pd
        print(f"✅ pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ numpy {np.__version__}")
    except ImportError as e:
        print(f"❌ numpy 导入失败: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ scikit-learn 导入失败: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm")
    except ImportError as e:
        print(f"❌ tqdm 导入失败: {e}")
        return False
    
    try:
        import openpyxl
        print(f"✅ openpyxl {openpyxl.__version__}")
    except ImportError as e:
        print(f"⚠️  openpyxl 导入失败: {e} (Excel文件支持可能受限)")
    
    return True

def test_project_structure():
    """测试项目结构"""
    print("\n🏗️  测试项目结构...")
    
    required_dirs = [
        'data',
        'data_loading',
        'hypothesis_testing',
        'hypothesis_testing/hypothesis_1',
        'hypothesis_testing/hypothesis_2',
        'hypothesis_testing/hypothesis_3'
    ]
    
    required_files = [
        'requirements.txt',
        'README.md',
        'data_loading/unified_data_loader.py',
        'hypothesis_testing/hypothesis_1/hypothesis_1.py',
        'hypothesis_testing/hypothesis_2/hypothesis_2.py',
        'hypothesis_testing/hypothesis_3/hypothesis_3.py'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录缺失: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"✅ 文件存在: {file_path}")
        else:
            print(f"❌ 文件缺失: {file_path}")
            all_good = False
    
    return all_good

def test_data_loader():
    """测试数据加载器"""
    print("\n📊 测试数据加载器...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data_loading.unified_data_loader import get_unified_data_loader
        
        loader = get_unified_data_loader()
        print("✅ 数据加载器初始化成功")
        
        # 测试数据摘要
        summary = loader.get_data_summary()
        print("✅ 数据摘要获取成功")
        print(f"   数据摘要: {summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        return False

def test_hypothesis_imports():
    """测试假设验证模块导入"""
    print("\n🎯 测试假设验证模块...")
    
    try:
        from hypothesis_testing.hypothesis_1.hypothesis_1 import Hypothesis1ValidatorCorrect
        print("✅ Hypothesis 1 导入成功")
    except Exception as e:
        print(f"❌ Hypothesis 1 导入失败: {e}")
        return False
    
    try:
        from hypothesis_testing.hypothesis_2.hypothesis_2 import Hypothesis2Validator
        print("✅ Hypothesis 2 导入成功")
    except Exception as e:
        print(f"❌ Hypothesis 2 导入失败: {e}")
        return False
    
    try:
        from hypothesis_testing.hypothesis_3.hypothesis_3 import Hypothesis3Validator
        print("✅ Hypothesis 3 导入成功")
    except Exception as e:
        print(f"❌ Hypothesis 3 导入失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始环境测试...")
    print("=" * 60)
    
    tests = [
        ("包导入测试", test_imports),
        ("项目结构测试", test_project_structure),
        ("数据加载器测试", test_data_loader),
        ("假设验证模块测试", test_hypothesis_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！环境配置正确。")
        return True
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
