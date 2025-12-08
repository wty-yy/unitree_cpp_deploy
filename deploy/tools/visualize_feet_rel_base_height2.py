import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import xml.etree.ElementTree as ET

# ==========================================
# 1. 运动学参数定义 (基于 go2.urdf 解析)
# ==========================================
# 由于环境限制未使用 pinocchio，这里使用从 URDF 提取的参数
# 关节顺序：Hip (Roll), Thigh (Pitch), Calf (Pitch)
# 偏移量 (xyz) 和 旋转轴
legs_params = {
    'FL': {'ids': [0, 1, 2], 'signs': [1, 1, 1], 'params': [
        {'xyz': [0.1934, 0.0465, 0.0], 'axis': 'x'}, # Hip
        {'xyz': [0.0, 0.0955, 0.0], 'axis': 'y'},    # Thigh
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'y'},    # Calf
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'fixed'} # Foot
    ]},
    'FR': {'ids': [3, 4, 5], 'signs': [1, 1, 1], 'params': [
        {'xyz': [0.1934, -0.0465, 0.0], 'axis': 'x'},
        {'xyz': [0.0, -0.0955, 0.0], 'axis': 'y'},
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'y'},
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'fixed'}
    ]},
    'RL': {'ids': [6, 7, 8], 'signs': [1, 1, 1], 'params': [
        {'xyz': [-0.1934, 0.0465, 0.0], 'axis': 'x'},
        {'xyz': [0.0, 0.0955, 0.0], 'axis': 'y'},
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'y'},
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'fixed'}
    ]},
    'RR': {'ids': [9, 10, 11], 'signs': [1, 1, 1], 'params': [
        {'xyz': [-0.1934, -0.0465, 0.0], 'axis': 'x'},
        {'xyz': [0.0, -0.0955, 0.0], 'axis': 'y'},
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'y'},
        {'xyz': [0.0, 0.0, -0.213], 'axis': 'fixed'}
    ]}
}

def get_transform_matrix(xyz, axis, q):
    # 平移矩阵
    T = np.eye(4)
    T[:3, 3] = xyz
    
    # 旋转矩阵
    c, s = np.cos(q), np.sin(q)
    R = np.eye(4)
    if axis == 'x':
        R = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    elif axis == 'y':
        R = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    # 'fixed' 关节没有旋转，保持单位矩阵
    
    return T @ R

def compute_foot_height(leg_name, q_all):
    leg = legs_params[leg_name]
    T_accum = np.eye(4)
    
    # 遍历该腿的每个关节 (Hip -> Thigh -> Calf -> Foot)
    for i, param in enumerate(leg['params']):
        # 获取关节角度，最后一段(foot)是固定的，角度为0
        q = q_all[leg['ids'][i]] if i < 3 else 0.0
        T_i = get_transform_matrix(param['xyz'], param['axis'], q)
        T_accum = T_accum @ T_i
        
    return T_accum[2, 3] # 返回 Z 坐标

# ==========================================
# 2. 数据处理与绘图
# ==========================================
def plot_foot_heights(csv_file, start_time=None, end_time=None):
    df = pd.read_csv(csv_file)
    
    # 时间过滤
    if start_time and end_time:
        # 将 wall_time 转换为 datetime 对象进行比较
        # 假设 wall_time 格式类似 "22:54:36.3"
        time_objs = pd.to_datetime(df['wall_time'], format='%H:%M:%S.%f').dt.time
        t_start = datetime.strptime(start_time, '%H:%M:%S').time()
        t_end = datetime.strptime(end_time, '%H:%M:%S').time()
        
        mask = (time_objs >= t_start) & (time_objs <= t_end)
        df_filtered = df[mask]
    else:
        df_filtered = df

    if df_filtered.empty:
        print("警告：筛选后数据为空，请检查时间区间。")
        return

    # 计算高度
    results = {'time': df_filtered['wall_time'], 'FL': [], 'FR': [], 'RL': [], 'RR': []}
    
    for _, row in df_filtered.iterrows():
        q = [row[f'q_{i}'] for i in range(12)]
        results['FL'].append(compute_foot_height('FL', q))
        results['FR'].append(compute_foot_height('FR', q))
        results['RL'].append(compute_foot_height('RL', q))
        results['RR'].append(compute_foot_height('RR', q))
    
    # 绘图
    plt.figure(figsize=(12, 6))
    x_axis = pd.to_datetime(results['time'], format='%H:%M:%S.%f')
    
    plt.plot(x_axis, results['FL'], label='FL (Front Left)')
    plt.plot(x_axis, results['FR'], label='FR (Front Right)')
    plt.plot(x_axis, results['RL'], label='RL (Rear Left)')
    plt.plot(x_axis, results['RR'], label='RR (Rear Right)')
    
    plt.title(f'Foot Vertical Height Relative to Base\nTime Range: {start_time} - {end_time}' if start_time else 'Foot Vertical Height Relative to Base')
    plt.xlabel('Time (Wall Time)')
    plt.ylabel('Vertical Height (m)')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. 执行分析
# ==========================================
csv_path = 'run_data_2025-12-08_22-54-32.csv'

# 示例：绘制特定时间段 (您可以修改此处的时间)
# start_time = "22:54:36"
# end_time = "22:54:37"
# plot_foot_heights(csv_path, start_time, end_time)

# 默认绘制所有数据
plot_foot_heights(csv_path)