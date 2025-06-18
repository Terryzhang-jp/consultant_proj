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

        # 适配真实数据的列名
        # 真实数据列名: LP, OFFICE, UNIT, MGR, AMGR, JOB_YYY, JOB_MM, JOB_DD, RANK, STATUS
        required_columns_mapping = {
            'LP': 'LP',
            'AMGR': 'AMGR',
            'OFFICE': 'OFFICE',
            'RANK': 'RANK_x',  # 映射RANK到RANK_x以保持兼容性
            'JOB_YYY': 'JOB_YYY',
            'JOB_MM': 'JOB_MM',
            'JOB_DD': 'JOB_DD'
        }

        # 检查必要列是否存在
        missing_columns = []
        for real_col, expected_col in required_columns_mapping.items():
            if real_col not in df.columns:
                missing_columns.append(real_col)

        if missing_columns:
            raise ValueError(f"LP历史数据缺少必要列: {missing_columns}")

        # 重命名列以保持兼容性
        df = df.rename(columns={'RANK': 'RANK_x'})

        # 转换年份格式 (2位数转4位数)
        # JOB_YYY=187 表示 1987年，需要加1900
        df['S_YR'] = df['JOB_YYY'] + 1900
        df['S_MO'] = df['JOB_MM']

        return df
    
    def load_reward_data(self) -> pd.DataFrame:
        """加载报酬数据"""
        file_path = os.path.join(self.data_dir, '報酬データ_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"报酬数据文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # 适配真实数据的列名
        # 真实数据列名: SYAIN_CODE, S_YR, S_MO, SASHIHIKI, RANK
        # 注意：真实数据没有KYUYO列，我们需要生成它
        required_columns = ['SYAIN_CODE', 'S_YR', 'S_MO', 'SASHIHIKI']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"报酬数据缺少必要列: {missing_columns}")

        # 统一列名
        df['LP'] = df['SYAIN_CODE']

        # 生成KYUYO列（假设KYUYO = SASHIHIKI * 1.1，这是一个合理的估算）
        df['KYUYO'] = df['SASHIHIKI'] * 1.1

        # 转换年份格式 (2位数转4位数)
        # S_YR=188 表示 1988年，需要加1900
        df['S_YR'] = df['S_YR'] + 1900
        
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

        # 适配真实数据格式
        # 真实数据列名: LP, ym, 年換算手数料, 件数
        # ym格式: 201609 (YYYYMM)
        if 'ym' in df.columns:
            # 从ym列提取年份和月份
            df['S_YR'] = df['ym'] // 100  # 201609 -> 2016
            df['S_MO'] = df['ym'] % 100   # 201609 -> 9

        return df
    
    def load_mtg_attendance_data(self) -> pd.DataFrame:
        """加载MTG出席率数据"""
        file_path = os.path.join(self.data_dir, 'MTG出席率2021-2023_hashed.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MTG出席率数据文件不存在: {file_path}")

        df = pd.read_csv(file_path)

        # 适配真实数据格式
        # 真实数据列名: LPコード, HONBUID, 開催日, 代替日, 廃止有無, 出欠, 欠席理由, 出席扱い
        # 统一列名
        df['LP'] = df['LPコード']

        # 从開催日提取年份和月份
        df['開催日'] = pd.to_datetime(df['開催日'], errors='coerce')
        df['S_YR'] = df['開催日'].dt.year
        df['S_MO'] = df['開催日'].dt.month

        return df
    
    def load_complaints_data(self) -> pd.DataFrame:
        """加载苦情数据"""
        file_path = os.path.join(self.data_dir, '苦情データ_hashed.xlsx')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"苦情数据文件不存在: {file_path}")

        # 真实数据的sheet名称是'Sheet1'
        df = pd.read_excel(file_path, sheet_name='Sheet1')
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

        # 真实数据的sheet名称是'T事務ミス台帳'
        df = pd.read_excel(file_path, sheet_name='T事務ミス台帳')
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
            # MTG数据已经在load函数中处理了S_YR和S_MO
            # 计算出席率
            mtg_summary = mtg_data.groupby(['LP', 'S_YR', 'S_MO']).agg({
                '出席扱い': 'mean'  # 计算平均出席率
            }).reset_index()
            mtg_summary = mtg_summary.rename(columns={'出席扱い': 'ATTENDANCE_RATE'})

            if final_df is not None:
                final_df = pd.merge(final_df, mtg_summary,
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
