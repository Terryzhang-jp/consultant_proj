#!/usr/bin/env python3
"""
统一测试数据生成脚本
为所有hypothesis生成全面的测试数据
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# 设置随机种子以确保可重现性
np.random.seed(42)
random.seed(42)

def generate_comprehensive_test_data():
    """生成全面的测试数据"""
    
    # 基础参数
    num_lps = 1000  # LP数量
    num_amgrs = 50  # AMGR数量
    num_offices = 10  # Office数量
    start_year = 2010
    end_year = 2024
    
    # 生成LP列表
    lps = [f"LP_{i:04d}" for i in range(1, num_lps + 1)]
    amgrs = [f"LP_{i:04d}" for i in range(8001, 8001 + num_amgrs)]
    offices = [f"OFFICE_{i:02d}" for i in range(1, num_offices + 1)]
    
    print("开始生成测试数据...")
    
    # 1. 生成LPヒストリー数据
    print("生成LPヒストリー数据...")
    lp_history_data = []
    
    for lp in lps:
        # 为每个LP分配一个主要的AMGR (80%的时间)
        primary_amgr = random.choice(amgrs)
        
        # 80%的LP是RANK_x=10
        rank = 10 if random.random() < 0.8 else random.choice([20, 30])
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 80%的时间使用主要AMGR，20%的时间可能变更
                current_amgr = primary_amgr if random.random() < 0.8 else random.choice(amgrs)
                current_office = random.choice(offices)
                
                lp_history_data.append({
                    'LP': lp,
                    'AMGR': current_amgr,
                    'OFFICE': current_office,
                    'RANK_x': rank,
                    'JOB_YYY': year - 2000,  # 转换为2位数年份
                    'JOB_MM': month,
                    'JOB_DD': random.randint(1, 28)
                })
    
    lp_history_df = pd.DataFrame(lp_history_data)
    
    # 2. 生成報酬データ (SASHIHIKI)
    print("生成報酬データ...")
    reward_data = []
    
    for lp in lps:
        # 为每个LP设定基础薪资水平
        if lp.endswith(('001', '002', '003', '004', '005')):
            # 前5个LP设为低业绩案例
            base_salary = random.randint(200000, 350000)
        else:
            base_salary = random.randint(300000, 600000)
        
        for year in range(start_year, end_year + 1):
            # 先生成正常的月度数据
            monthly_data = []
            for month in range(1, 13):
                # 添加月度波动
                monthly_variation = random.uniform(0.7, 1.3)
                kyuyo = int(base_salary * monthly_variation)
                sashihiki = int(kyuyo * random.uniform(0.75, 0.95))

                monthly_data.append({
                    'SYAIN_CODE': lp,
                    'S_YR': year - 2000,  # 转换为2位数年份
                    'S_MO': month,
                    'KYUYO': kyuyo,
                    'SASHIHIKI': sashihiki
                })

            # 为hypothesis 2创建收入集中案例
            if random.random() < 0.2:  # 20%的LP-年度组合有收入集中
                # 随机选择2个月，让它们占年收入的60%以上
                high_months = random.sample(range(12), 2)

                # 计算原始年度总收入
                original_total = sum([data['SASHIHIKI'] for data in monthly_data])

                # 设定高集中度目标（60%）
                target_concentration = 0.6
                high_total = int(original_total * target_concentration)
                low_total = original_total - high_total

                # 更新选中的2个月为高收入
                for month_idx in high_months:
                    monthly_data[month_idx]['SASHIHIKI'] = high_total // 2
                    monthly_data[month_idx]['KYUYO'] = int((high_total // 2) * 1.1)

                # 调整其他月份的收入
                other_months = [i for i in range(12) if i not in high_months]
                if other_months:
                    avg_low = low_total // len(other_months)
                    for month_idx in other_months:
                        monthly_data[month_idx]['SASHIHIKI'] = max(avg_low, 50000)  # 最低5万
                        monthly_data[month_idx]['KYUYO'] = int(monthly_data[month_idx]['SASHIHIKI'] * 1.1)

            # 添加到总数据中
            reward_data.extend(monthly_data)
    
    reward_df = pd.DataFrame(reward_data)
    
    # 3. 生成懲戒処分データ
    print("生成懲戒処分データ...")
    discipline_data = []
    
    # 选择一些LP进行懲戒処分
    discipline_lps = random.sample(lps, min(200, len(lps)))  # 20%的LP有懲戒処分记录
    discipline_types = ['警告', '注意', '厳重注意']
    
    for lp in discipline_lps:
        # 每个LP可能有1-3次懲戒処分
        num_disciplines = random.randint(1, 3)
        
        for _ in range(num_disciplines):
            # 随机选择年份和日期
            year = random.randint(2015, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            
            discipline_data.append({
                'LP NO': lp,
                'FUKABI': f"{year}/{month:02d}/{day:02d}",
                'SHOBUN': random.choice(discipline_types)
            })
    
    discipline_df = pd.DataFrame(discipline_data)
    
    # 4. 生成業績データ (用于hypothesis 3等)
    print("生成業績データ...")
    performance_data = []
    
    for lp in lps:
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 生成各种业绩指标
                performance_data.append({
                    'LP': lp,
                    'S_YR': year - 2000,
                    'S_MO': month,
                    'SEISEKI_1': random.randint(0, 100),
                    'SEISEKI_2': random.randint(0, 50),
                    'SEISEKI_3': random.randint(0, 200),
                    'TOTAL_SEISEKI': random.randint(50, 500)
                })
    
    performance_df = pd.DataFrame(performance_data)
    
    # 5. 生成其他辅助数据
    print("生成其他辅助数据...")
    
    # MTG出席率データ
    mtg_data = []
    for lp in lps:
        for year in range(2021, 2024):
            for month in range(1, 13):
                mtg_data.append({
                    'LP': lp,
                    'YEAR': year,
                    'MONTH': month,
                    'ATTENDANCE_RATE': random.uniform(0.5, 1.0)
                })
    
    mtg_df = pd.DataFrame(mtg_data)
    
    # 苦情データ
    complaint_data = []
    complaint_lps = random.sample(lps, min(100, len(lps)))
    
    for lp in complaint_lps:
        num_complaints = random.randint(1, 5)
        for _ in range(num_complaints):
            year = random.randint(2020, 2024)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            
            complaint_data.append({
                'LP': lp,
                'DATE': f"{year}/{month:02d}/{day:02d}",
                'COMPLAINT_TYPE': random.choice(['苦情A', '苦情B', '苦情C']),
                'SEVERITY': random.choice(['軽微', '中程度', '重大'])
            })
    
    complaint_df = pd.DataFrame(complaint_data)
    
    # 社長杯入賞履歴
    award_data = []
    award_lps = random.sample(lps, min(50, len(lps)))
    
    for lp in award_lps:
        num_awards = random.randint(1, 3)
        for _ in range(num_awards):
            year = random.randint(2015, 2024)
            award_data.append({
                'LP': lp,
                'YEAR': year,
                'AWARD_TYPE': random.choice(['金賞', '銀賞', '銅賞']),
                'RANK': random.randint(1, 100)
            })
    
    award_df = pd.DataFrame(award_data)
    
    return {
        'lp_history': lp_history_df,
        'reward': reward_df,
        'discipline': discipline_df,
        'performance': performance_df,
        'mtg_attendance': mtg_df,
        'complaints': complaint_df,
        'awards': award_df
    }

def save_data_to_files(data_dict, output_dir):
    """将数据保存到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存各个数据文件
    data_dict['lp_history'].to_csv(os.path.join(output_dir, 'LPヒストリー_hashed.csv'), index=False)
    data_dict['reward'].to_csv(os.path.join(output_dir, '報酬データ_hashed.csv'), index=False)
    data_dict['discipline'].to_csv(os.path.join(output_dir, '懲戒処分_事故区分等追加_hashed.csv'), index=False)
    data_dict['performance'].to_csv(os.path.join(output_dir, '業績_hashed.csv'), index=False)
    data_dict['mtg_attendance'].to_csv(os.path.join(output_dir, 'MTG出席率2021-2023_hashed.csv'), index=False)
    data_dict['awards'].to_csv(os.path.join(output_dir, '社長杯入賞履歴_LPコード0埋_hashed.csv'), index=False)
    
    # 保存苦情データ为Excel格式
    with pd.ExcelWriter(os.path.join(output_dir, '苦情データ_hashed.xlsx')) as writer:
        data_dict['complaints'].to_excel(writer, sheet_name='苦情データ', index=False)
    
    # 保存事務ミスデータ (简化版)
    office_error_data = data_dict['complaints'].copy()
    office_error_data['ERROR_TYPE'] = office_error_data['COMPLAINT_TYPE'].str.replace('苦情', '事務ミス')
    
    with pd.ExcelWriter(os.path.join(output_dir, '★事務ミスデータ_不要データ削除版_hashed.xlsx')) as writer:
        office_error_data.to_excel(writer, sheet_name='事務ミス', index=False)
    
    print(f"数据已保存到: {output_dir}")
    print(f"生成的文件:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    # 生成数据
    data = generate_comprehensive_test_data()
    
    # 保存到data文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = current_dir
    
    save_data_to_files(data, output_dir)
    
    print("\n数据生成完成!")
    print(f"数据统计:")
    print(f"  - LP数量: {len(data['lp_history']['LP'].unique())}")
    print(f"  - AMGR数量: {len(data['lp_history']['AMGR'].unique())}")
    print(f"  - 報酬データ记录数: {len(data['reward'])}")
    print(f"  - 懲戒処分记录数: {len(data['discipline'])}")
    print(f"  - 業績データ记录数: {len(data['performance'])}")
