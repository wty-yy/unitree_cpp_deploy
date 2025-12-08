import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_odom(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check for required columns
    required_cols = ['odom_pos_0', 'odom_pos_1', 'odom_pos_2']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Odometry Visualization: {os.path.basename(csv_path)}', fontsize=16)
    
    # 1. 3D Trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(df['odom_pos_0'], df['odom_pos_1'], df['odom_pos_2'], label='Trajectory', linewidth=2, color='purple')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # 2. X-Y Plane (Top View)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df['odom_pos_0'], df['odom_pos_1'], label='Path', linewidth=2, color='orange')
    ax2.scatter(df['odom_pos_0'].iloc[0], df['odom_pos_1'].iloc[0], color='green', marker='o', label='Start', zorder=5)
    ax2.scatter(df['odom_pos_0'].iloc[-1], df['odom_pos_1'].iloc[-1], color='red', marker='x', label='End', zorder=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('2D Path (X-Y Plane)')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    # 3. Position vs Time
    ax3 = fig.add_subplot(2, 1, 2)
    
    if 'time' in df.columns:
        x_axis = df['time']
        x_label = 'Time (s)'
    else:
        x_axis = df.index
        x_label = 'Step'
    
    ax3.plot(x_axis, df['odom_pos_0'], label='X Position', linewidth=1.5, alpha=0.8)
    ax3.plot(x_axis, df['odom_pos_1'], label='Y Position', linewidth=1.5, alpha=0.8)
    ax3.plot(x_axis, df['odom_pos_2'], label='Z Position', linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel(x_label)
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position Components over Time')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Odometry Position from CSV")
    parser.add_argument("csv_file", type=str, help="Path to the CSV log file")
    args = parser.parse_args()

    plot_odom(args.csv_file)
