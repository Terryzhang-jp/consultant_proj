#!/usr/bin/env python3
"""
统一数据加载器
为所有hypothesis提供统一的数据加载接口
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional, List

class UnifiedDataLoader:
    """统一数据加载器类"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径，默认为项目根目录下的data文件夹
        """
        if data_dir is None:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            self.data_dir = os.path.join(project_root, 'data')
        else:
            self.data_dir = data_dir
            
        # 验证数据目录存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
    
    def load_lp_history(self) -> pd.DataFrame:
        """加载LP历史数据"""
        file_path = os.path.join(self.data_dir, 'LPヒストリー_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"LP历史数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 数据类型转换和验证
        required_columns = ['LP', 'AMGR', 'OFFICE', 'RANK_x', 'JOB_YYY', 'JOB_MM', 'JOB_DD']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"LP历史数据缺少必要列: {missing_columns}")
        
        # 转换年份格式 (2位数转4位数)
        df['S_YR'] = df['JOB_YYY'] + 2000
        df['S_MO'] = df['JOB_MM']
        
        return df
    
    def load_reward_data(self) -> pd.DataFrame:
        """加载报酬数据"""
        file_path = os.path.join(self.data_dir, '報酬データ_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"报酬数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 数据类型转换和验证
        required_columns = ['SYAIN_CODE', 'S_YR', 'S_MO', 'KYUYO', 'SASHIHIKI']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"报酬数据缺少必要列: {missing_columns}")
        
        # 统一列名
        df['LP'] = df['SYAIN_CODE']
        
        # 转换年份格式 (2位数转4位数)
        df['S_YR'] = df['S_YR'] + 2000
        
        return df
    
    def load_discipline_data(self) -> pd.DataFrame:
        """加载懲戒処分数据"""
        file_path = os.path.join(self.data_dir, '懲戒処分_事故区分等追加_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"懲戒処分数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 数据类型转换和验证
        required_columns = ['LP NO', 'FUKABI', 'SHOBUN']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"懲戒処分数据缺少必要列: {missing_columns}")
        
        # 统一列名
        df['LP'] = df['LP NO']
        
        # 处理日期格式
        df['FUKABI'] = pd.to_datetime(df['FUKABI'], format='%Y/%m/%d', errors='coerce')
        df['S_YR'] = df['FUKABI'].dt.year
        df['S_MO'] = df['FUKABI'].dt.month
        
        return df
    
    def load_performance_data(self) -> pd.DataFrame:
        """加载業績数据"""
        file_path = os.path.join(self.data_dir, '業績_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"業績数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 转换年份格式 (2位数转4位数)
        if 'S_YR' in df.columns:
            df['S_YR'] = df['S_YR'] + 2000
        
        return df
    
    def load_mtg_attendance_data(self) -> pd.DataFrame:
        """加载MTG出席率数据"""
        file_path = os.path.join(self.data_dir, 'MTG出席率2021-2023_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MTG出席率数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_complaints_data(self) -> pd.DataFrame:
        """加载苦情数据"""
        file_path = os.path.join(self.data_dir, '苦情データ_hashed.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"苦情数据文件不存在: {file_path}")
        
        df = pd.read_excel(file_path, sheet_name='苦情データ')
        return df
    
    def load_awards_data(self) -> pd.DataFrame:
        """加载社長杯入賞履歴数据"""
        file_path = os.path.join(self.data_dir, '社長杯入賞履歴_LPコード0埋_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"社長杯入賞履歴数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        return df
    
    def load_office_errors_data(self) -> pd.DataFrame:
        """加载事務ミス数据"""
        file_path = os.path.join(self.data_dir, '★事務ミスデータ_不要データ削除版_hashed.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"事務ミス数据文件不存在: {file_path}")
        
        df = pd.read_excel(file_path, sheet_name='事務ミス')
        return df
    
    def create_final_dataset(self, 
                           include_lp_history: bool = True,
                           include_reward: bool = True,
                           include_discipline: bool = True,
                           include_performance: bool = False,
                           include_mtg: bool = False,
                           include_complaints: bool = False,
                           include_awards: bool = False,
                           include_office_errors: bool = False) -> pd.DataFrame:
        """
        创建最终数据集，合并多个数据源
        
        Args:
            include_*: 是否包含各种数据源
            
        Returns:
            合并后的数据集
        """
        final_df = None
        
        # 加载基础数据
        if include_lp_history:
            lp_history = self.load_lp_history()
            if final_df is None:
                final_df = lp_history
            else:
                final_df = pd.merge(final_df, lp_history, 
                                  on=['LP', 'S_YR', 'S_MO'], how='outer')
        
        if include_reward:
            reward_data = self.load_reward_data()
            if final_df is None:
                final_df = reward_data
            else:
                final_df = pd.merge(final_df, reward_data[['LP', 'S_YR', 'S_MO', 'KYUYO', 'SASHIHIKI']], 
                                  on=['LP', 'S_YR', 'S_MO'], how='outer')
        
        if include_discipline:
            discipline_data = self.load_discipline_data()
            # 为懲戒処分数据添加标记
            discipline_data['SHOBUN_FLAG'] = 1
            discipline_summary = discipline_data.groupby(['LP', 'S_YR', 'S_MO']).agg({
                'SHOBUN': 'first',
                'SHOBUN_FLAG': 'sum'
            }).reset_index()
            
            if final_df is None:
                final_df = discipline_summary
            else:
                final_df = pd.merge(final_df, discipline_summary, 
                                  on=['LP', 'S_YR', 'S_MO'], how='left')
        
        # 加载其他可选数据
        if include_performance:
            performance_data = self.load_performance_data()
            if final_df is not None:
                final_df = pd.merge(final_df, performance_data, 
                                  on=['LP', 'S_YR', 'S_MO'], how='left')
        
        if include_mtg:
            mtg_data = self.load_mtg_attendance_data()
            # 转换MTG数据格式以匹配
            mtg_data['S_YR'] = mtg_data['YEAR']
            mtg_data['S_MO'] = mtg_data['MONTH']
            if final_df is not None:
                final_df = pd.merge(final_df, mtg_data[['LP', 'S_YR', 'S_MO', 'ATTENDANCE_RATE']], 
                                  on=['LP', 'S_YR', 'S_MO'], how='left')
        
        # 填充缺失值
        if final_df is not None:
            final_df['SHOBUN_FLAG'] = final_df['SHOBUN_FLAG'].fillna(0)
            
        return final_df
    
    def get_data_summary(self) -> Dict[str, int]:
        """获取数据摘要信息"""
        summary = {}
        
        try:
            lp_history = self.load_lp_history()
            summary['LP数量'] = lp_history['LP'].nunique()
            summary['AMGR数量'] = lp_history['AMGR'].nunique()
            summary['LP历史记录数'] = len(lp_history)
        except Exception as e:
            summary['LP历史数据'] = f"加载失败: {str(e)}"
        
        try:
            reward_data = self.load_reward_data()
            summary['报酬记录数'] = len(reward_data)
            summary['年份范围'] = f"{reward_data['S_YR'].min()}-{reward_data['S_YR'].max()}"
        except Exception as e:
            summary['报酬数据'] = f"加载失败: {str(e)}"
        
        try:
            discipline_data = self.load_discipline_data()
            summary['懲戒処分记录数'] = len(discipline_data)
            summary['懲戒処分LP数'] = discipline_data['LP'].nunique()
        except Exception as e:
            summary['懲戒処分数据'] = f"加载失败: {str(e)}"
        
        return summary

# 创建全局数据加载器实例
data_loader = UnifiedDataLoader()

def get_unified_data_loader() -> UnifiedDataLoader:
    """获取统一数据加载器实例"""
    return data_loader
