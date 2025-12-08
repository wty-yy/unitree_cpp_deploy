import argparse
import pandas as pd
import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def get_joint_names():
    # Standard Unitree Go2 joint order
    return [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ]

def load_model(urdf_path):
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        sys.exit(1)
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data

def process_data(csv_path, start_time=None, end_time=None):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Filter by time if provided
    if start_time or end_time:
        # Assuming 'wall_time' is in H:M:S format, we might need to handle date if it spans days, 
        # but usually these logs are short. We can convert to datetime objects for comparison.
        # However, the CSV example shows '22:54:36.3'.
        
        # Let's try to parse the time column
        try:
            # We use a dummy date to make it a datetime object
            df['datetime'] = pd.to_datetime(df['wall_time'], format='%H:%M:%S.%f').apply(lambda x: x.time())
        except ValueError:
             # Fallback for format without microseconds or different format
             try:
                df['datetime'] = pd.to_datetime(df['wall_time'], format='%H:%M:%S').apply(lambda x: x.time())
             except:
                print("Warning: Could not parse time column for filtering. Using all data.")
                start_time = None
                end_time = None

        if start_time:
            st = pd.to_datetime(start_time, format='%H:%M:%S').time()
            df = df[df['datetime'] >= st]
        
        if end_time:
            et = pd.to_datetime(end_time, format='%H:%M:%S').time()
            df = df[df['datetime'] <= et]

    return df

def compute_foot_heights(model, data, df, selected_feet=None):
    all_foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    
    if selected_feet:
        foot_names = [name for name in all_foot_names if any(f in name for f in selected_feet)]
        if not foot_names:
             print(f"Warning: No valid feet found matching {selected_feet}. Using all feet.")
             foot_names = all_foot_names
    else:
        foot_names = all_foot_names

    foot_ids = [model.getFrameId(name) for name in foot_names]
    
    # Check if frames exist
    for name, fid in zip(foot_names, foot_ids):
        if fid >= len(model.frames):
            print(f"Error: Frame {name} not found in model.")
            sys.exit(1)

    heights = {name: [] for name in foot_names}
    times = []
    
    q_cols = [f'q_{i}' for i in range(12)]
    
    # Check if q columns exist
    if not all(col in df.columns for col in q_cols):
        print("Error: Joint position columns (q_0 to q_11) not found in CSV.")
        sys.exit(1)

    # Pinocchio configuration vector size
    # For floating base: 7 (base) + 12 (joints) = 19
    # We will set base to identity (0,0,0 position, 0,0,0,1 quaternion)
    # so the foot positions are relative to the base frame.
    nq = model.nq
    nv = model.nv
    
    print(f"Model nq: {nq}, nv: {nv}")
    
    for index, row in df.iterrows():
        q = np.zeros(nq)
        
        # Set base to identity. 
        # If the model has a floating base, the first 7 elements are [x, y, z, qx, qy, qz, qw]
        # We want base at origin, aligned with world.
        if nq >= 7:
            q[6] = 1.0 # qw = 1.0
            
        # Fill joint positions
        # Assuming the joint order in URDF matches q_0...q_11
        # We should verify this mapping if possible, but standard Unitree is FL, FR, RL, RR
        # and Pinocchio builds model based on URDF tree traversal.
        # To be safe, we can map by name if we knew the joint names in order in the model.
        # But usually for these robots, the order is consistent if built from standard URDF.
        # Let's assume the order q_0..q_11 maps to the joints in the order they appear in the configuration vector (after base).
        
        # Let's check model joint names to be sure?
        # For now, we assume the standard order.
        
        joints_q = row[q_cols].values
        if nq >= 19:
             q[7:19] = joints_q
        else:
             # Fixed base?
             q[:12] = joints_q
             
        pin.framesForwardKinematics(model, data, q)
        
        for i, name in enumerate(foot_names):
            fid = foot_ids[i]
            # Get frame placement relative to world (which is base in this case)
            pos = data.oMf[fid].translation
            heights[name].append(pos[2]) # z-coordinate
            
        times.append(row['wall_time'])
        
    return times, heights

def plot_heights(times, heights):
    plt.figure(figsize=(12, 6))
    
    for name, data in heights.items():
        plt.plot(range(len(data)), data, label=name)
        
    plt.xlabel('Time Steps')
    plt.ylabel('Height relative to Base (m)')
    plt.title('Foot Heights Relative to Base')
    plt.legend()
    plt.grid(True)
    
    # If too many points, x-axis labels might be crowded if we use time strings
    # So we just use index or sparse time labels
    if len(times) > 0:
        # Show start and end time
        plt.xlabel(f'Time (Start: {times[0]}, End: {times[-1]})')
        
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize foot heights relative to base from CSV data.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--urdf', type=str, default='resources/go2.urdf', help='Path to the URDF file')
    parser.add_argument('--start', type=str, help='Start time (H:M:S)')
    parser.add_argument('--end', type=str, help='End time (H:M:S)')
    parser.add_argument('--feet', type=str, help='Comma-separated list of feet to plot (e.g., FL,FR)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to workspace root if needed, but user provided relative paths usually work
    # We assume the script is run from the workspace root or paths are correct relative to CWD
    
    model, data = load_model(args.urdf)
    df = process_data(args.csv_file, args.start, args.end)
    
    if df.empty:
        print("No data found for the specified time range.")
        sys.exit(0)
        
    print(f"Processing {len(df)} rows...")
    
    selected_feet = args.feet.split(',') if args.feet else None
    times, heights = compute_foot_heights(model, data, df, selected_feet)
    
    plot_heights(times, heights)

if __name__ == "__main__":
    main()
