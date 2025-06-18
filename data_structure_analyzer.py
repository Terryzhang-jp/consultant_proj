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
            "row_count": len(df)
        }

        # 列的数据类型（只保留关键信息）
        dtypes = {}
        for col in df.columns:
            dtypes[col] = str(df[col].dtype)
        info["column_dtypes"] = dtypes

        # 只统计缺失值比例，不详细统计
        missing_summary = {}
        for col in df.columns:
            missing_pct = round(df[col].isnull().sum() / len(df) * 100, 2)
            if missing_pct > 0:  # 只记录有缺失值的列
                missing_summary[col] = missing_pct
        info["missing_percentage"] = missing_summary

        # 只保留前2行样本数据
        try:
            sample_data = df.head(2).to_dict('records')
            # 简化数据转换
            for record in sample_data:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value) if isinstance(value, np.floating) else int(value)
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

                # 只统计有缺失值的列
                missing_summary = {}
                for col in df.columns:
                    missing_pct = round(df[col].isnull().sum() / len(df) * 100, 2)
                    if missing_pct > 0:
                        missing_summary[col] = missing_pct
                sheet_info["missing_percentage"] = missing_summary

                # 只保留前2行样本数据
                try:
                    sample_data = df.head(2).to_dict('records')
                    # 简化数据转换
                    for record in sample_data:
                        for key, value in record.items():
                            if pd.isna(value):
                                record[key] = None
                            elif isinstance(value, (np.integer, np.floating)):
                                record[key] = float(value) if isinstance(value, np.floating) else int(value)
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
    """分析data目录中的关键文件"""
    if not os.path.exists(data_dir):
        return {"error": f"目录不存在: {data_dir}"}

    # 定义我们关心的关键文件（用于hypothesis验证）
    key_files = [
        "LPヒストリー_hashed.csv",
        "報酬データ_hashed.csv",
        "懲戒処分_事故区分等追加_hashed.csv",
        "業績_hashed.csv",
        "MTG出席率2021-2023_hashed.csv",
        "社長杯入賞履歴_LP",  # 部分匹配
        "苦情データ_hashed.xlsx",
        "事務ミスデータ"  # 部分匹配
    ]

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
    all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    # 过滤出关键文件
    files = []
    for file_name in all_files:
        for key_file in key_files:
            if key_file in file_name:
                files.append(file_name)
                break

    analysis_results["summary"]["total_files"] = len(files)
    analysis_results["summary"]["total_files_in_directory"] = len(all_files)
    
    print(f"🔍 开始分析 {data_dir} 目录中的 {len(files)} 个关键文件 (总共{len(all_files)}个文件)...")

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
