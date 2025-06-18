#!/usr/bin/env python3
"""
数据结构分析脚本
用于检查data文件夹中所有数据文件的列名、数据类型、形状等信息
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def analyze_csv_file(file_path):
    """分析CSV文件"""
    try:
        # 尝试不同的编码
        encodings = ['utf-8', 'shift_jis', 'cp932', 'gbk', 'latin1']
        df = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            return {"error": "无法读取文件 - 编码问题"}
        
        # 基本信息
        info = {
            "file_type": "CSV",
            "encoding": used_encoding,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "column_count": len(df.columns),
            "row_count": len(df),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # 列的数据类型
        dtypes = {}
        for col in df.columns:
            dtypes[col] = str(df[col].dtype)
        info["column_dtypes"] = dtypes
        
        # 缺失值统计
        missing_values = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = round(missing_count / len(df) * 100, 2)
            missing_values[col] = {
                "missing_count": int(missing_count),
                "missing_percentage": missing_pct
            }
        info["missing_values"] = missing_values
        
        # 数值列的基本统计
        numeric_stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                numeric_stats[col] = {
                    "min": float(stats['min']) if not pd.isna(stats['min']) else None,
                    "max": float(stats['max']) if not pd.isna(stats['max']) else None,
                    "mean": float(stats['mean']) if not pd.isna(stats['mean']) else None,
                    "std": float(stats['std']) if not pd.isna(stats['std']) else None,
                    "unique_count": int(df[col].nunique())
                }
            except:
                numeric_stats[col] = {"error": "统计计算失败"}
        info["numeric_statistics"] = numeric_stats
        
        # 文本列的基本信息
        text_stats = {}
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            try:
                unique_count = df[col].nunique()
                sample_values = df[col].dropna().head(5).tolist()
                text_stats[col] = {
                    "unique_count": int(unique_count),
                    "sample_values": sample_values
                }
            except:
                text_stats[col] = {"error": "统计计算失败"}
        info["text_statistics"] = text_stats
        
        # 前5行数据样本
        try:
            sample_data = df.head(5).to_dict('records')
            # 转换numpy类型为Python原生类型
            for record in sample_data:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, np.bool_):
                        record[key] = bool(value)
            info["sample_data"] = sample_data
        except:
            info["sample_data"] = "无法获取样本数据"
        
        return info
        
    except Exception as e:
        return {"error": f"分析失败: {str(e)}"}

def analyze_excel_file(file_path):
    """分析Excel文件"""
    try:
        # 获取所有sheet名称
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        info = {
            "file_type": "Excel",
            "sheet_names": sheet_names,
            "sheet_count": len(sheet_names),
            "sheets_info": {}
        }
        
        # 分析每个sheet
        for sheet_name in sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                sheet_info = {
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "column_count": len(df.columns),
                    "row_count": len(df)
                }
                
                # 列的数据类型
                dtypes = {}
                for col in df.columns:
                    dtypes[col] = str(df[col].dtype)
                sheet_info["column_dtypes"] = dtypes
                
                # 缺失值统计
                missing_values = {}
                for col in df.columns:
                    missing_count = df[col].isnull().sum()
                    missing_pct = round(missing_count / len(df) * 100, 2)
                    missing_values[col] = {
                        "missing_count": int(missing_count),
                        "missing_percentage": missing_pct
                    }
                sheet_info["missing_values"] = missing_values
                
                # 前3行数据样本
                try:
                    sample_data = df.head(3).to_dict('records')
                    # 转换numpy类型
                    for record in sample_data:
                        for key, value in record.items():
                            if pd.isna(value):
                                record[key] = None
                            elif isinstance(value, (np.integer, np.floating)):
                                record[key] = float(value) if isinstance(value, np.floating) else int(value)
                            elif isinstance(value, np.bool_):
                                record[key] = bool(value)
                    sheet_info["sample_data"] = sample_data
                except:
                    sheet_info["sample_data"] = "无法获取样本数据"
                
                info["sheets_info"][sheet_name] = sheet_info
                
            except Exception as e:
                info["sheets_info"][sheet_name] = {"error": f"分析失败: {str(e)}"}
        
        return info
        
    except Exception as e:
        return {"error": f"分析失败: {str(e)}"}

def analyze_data_directory(data_dir="data"):
    """分析data目录中的所有文件"""
    if not os.path.exists(data_dir):
        return {"error": f"目录不存在: {data_dir}"}
    
    analysis_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "data_directory": data_dir,
        "files_analyzed": {},
        "summary": {
            "total_files": 0,
            "csv_files": 0,
            "excel_files": 0,
            "other_files": 0,
            "failed_files": 0
        }
    }
    
    # 获取所有文件
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    analysis_results["summary"]["total_files"] = len(files)
    
    print(f"🔍 开始分析 {data_dir} 目录中的 {len(files)} 个文件...")
    
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        print(f"📄 分析文件: {file_name}")
        
        try:
            if file_name.endswith('.csv'):
                analysis_results["summary"]["csv_files"] += 1
                result = analyze_csv_file(file_path)
            elif file_name.endswith(('.xlsx', '.xls')):
                analysis_results["summary"]["excel_files"] += 1
                result = analyze_excel_file(file_path)
            else:
                analysis_results["summary"]["other_files"] += 1
                result = {"file_type": "其他", "note": "跳过分析"}
            
            if "error" in result:
                analysis_results["summary"]["failed_files"] += 1
                print(f"   ❌ 分析失败: {result['error']}")
            else:
                print(f"   ✅ 分析完成: {result.get('shape', 'N/A')} 形状")
            
            analysis_results["files_analyzed"][file_name] = result
            
        except Exception as e:
            analysis_results["summary"]["failed_files"] += 1
            analysis_results["files_analyzed"][file_name] = {"error": f"未知错误: {str(e)}"}
            print(f"   ❌ 分析失败: {str(e)}")
    
    return analysis_results

def save_analysis_results(results, output_file="data_structure_analysis.json"):
    """保存分析结果到JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📁 分析结果已保存到: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")
        return False

def print_summary(results):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("📊 数据结构分析摘要")
    print("="*60)
    
    summary = results.get("summary", {})
    print(f"总文件数: {summary.get('total_files', 0)}")
    print(f"CSV文件: {summary.get('csv_files', 0)}")
    print(f"Excel文件: {summary.get('excel_files', 0)}")
    print(f"其他文件: {summary.get('other_files', 0)}")
    print(f"分析失败: {summary.get('failed_files', 0)}")
    
    print("\n📋 文件详情:")
    for file_name, file_info in results.get("files_analyzed", {}).items():
        if "error" in file_info:
            print(f"  ❌ {file_name}: {file_info['error']}")
        else:
            shape = file_info.get("shape", "N/A")
            file_type = file_info.get("file_type", "未知")
            print(f"  ✅ {file_name}: {file_type}, 形状: {shape}")
            
            # 显示列名
            if "columns" in file_info:
                columns = file_info["columns"]
                print(f"     列数: {len(columns)}")
                print(f"     列名: {', '.join(columns[:10])}{'...' if len(columns) > 10 else ''}")

def main():
    """主函数"""
    print("🚀 开始数据结构分析...")
    
    # 分析data目录
    results = analyze_data_directory("data")
    
    # 保存结果
    save_analysis_results(results)
    
    # 打印摘要
    print_summary(results)
    
    print("\n🎯 分析完成！")
    print("请将 data_structure_analysis.json 文件内容发送给开发者进行数据结构适配。")

if __name__ == "__main__":
    main()
