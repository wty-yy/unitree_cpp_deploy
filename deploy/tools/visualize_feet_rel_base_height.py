"""
Visualize foot heights relative to base from CSV data with contact status.

Dependencies:
    conda install pinocchio pandas matplotlib

Usage:
    python visualize_feet_rel_base_height.py <path_to_csv> --urdf <path_to_urdf> [options]

Example:
    python visualize_feet_rel_base_height.py logs/data.csv --urdf resources/go2.urdf --force-threshold 10
"""

import argparse
import os
import sys
from datetime import datetime, time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pinocchio as pin

# --- Configuration & Constants ---
PATH_DIR = Path(__file__).parent.resolve()
PATH_IMGS = PATH_DIR / 'images'
PATH_IMGS.mkdir(exist_ok=True)

# Standard Unitree Go2 joint order
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

ALL_FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


# --- Helper Functions ---

def parse_time_str(time_str: str) -> Optional[time]:
    """Helper to parse time strings with variable precision."""
    if not time_str:
        return None
    
    formats = ['%H:%M:%S.%f', '%H:%M:%S']
    for fmt in formats:
        try:
            return pd.to_datetime(time_str, format=fmt).time()
        except ValueError:
            continue
    
    print(f"Warning: Could not parse time '{time_str}'. Expected format: H:M:S or H:M:S.ff")
    return None


def load_model(urdf_path: str) -> Tuple[pin.Model, pin.Data]:
    """Load the Pinocchio model from URDF."""
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        sys.exit(1)
    
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def process_data(csv_path: str, start_time: str = None, end_time: str = None) -> pd.DataFrame:
    """Load CSV data and filter by time range."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Pre-parse time column for filtering
    try:
        # Try converting to datetime objects first
        temp_times = pd.to_datetime(df['wall_time'], format='%H:%M:%S.%f', errors='coerce')
        # Fill failures (if any) with simplified format
        mask_nat = temp_times.isna()
        if mask_nat.any():
            temp_times[mask_nat] = pd.to_datetime(df.loc[mask_nat, 'wall_time'], format='%H:%M:%S', errors='coerce')
        
        df['datetime_obj'] = temp_times.dt.time
    except Exception as e:
        print(f"Warning: Could not parse 'wall_time' column ({e}). Using all data.")
        return df

    # Apply Time Filtering
    if start_time:
        st = parse_time_str(start_time)
        if st:
            df = df[df['datetime_obj'] >= st]
    
    if end_time:
        et = parse_time_str(end_time)
        if et:
            df = df[df['datetime_obj'] <= et]

    return df.reset_index(drop=True)


def compute_foot_heights(model: pin.Model, 
                         data: pin.Data, 
                         df: pd.DataFrame, 
                         selected_feet: List[str] = None) -> Tuple[List[str], Dict[str, List[float]]]:
    """Calculate foot heights relative to the base frame."""
    
    # Filter selected feet
    if selected_feet:
        clean_selected = [foot.strip().upper() for foot in selected_feet]
        foot_names = [name for name in ALL_FOOT_NAMES if any(name.startswith(s) for s in clean_selected)]
        if not foot_names:
            print(f"Warning: No matching feet for {selected_feet}. Using all feet.")
            foot_names = ALL_FOOT_NAMES
    else:
        foot_names = ALL_FOOT_NAMES

    # Get Frame IDs
    try:
        foot_ids = [model.getFrameId(name) for name in foot_names]
        # Validate Frame IDs
        for name, fid in zip(foot_names, foot_ids):
            if fid >= len(model.frames):
                raise ValueError(f"Frame {name} not found in model.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize storage
    heights = {name: [] for name in foot_names}
    times = df['wall_time'].tolist()
    
    # Check for joint columns
    q_cols = [f'q_{i}' for i in range(12)]
    if not all(col in df.columns for col in q_cols):
        print("Error: Joint position columns (q_0 to q_11) not found in CSV.")
        sys.exit(1)

    nq, nv = model.nq, model.nv
    print(f"Model nq: {nq}, nv: {nv}")

    # Main Kinematics Loop
    for _, row in df.iterrows():
        q = np.zeros(nq)

        # Set base to identity (0,0,0 pos, 0,0,0,1 quat) for relative calculation
        if nq >= 7:
            q[6] = 1.0 

        # Fill joint positions
        joints_q = row[q_cols].values
        if nq >= 19:
            q[7:19] = joints_q  # Floating base
        else:
            q[:12] = joints_q   # Fixed base

        pin.framesForwardKinematics(model, data, q)

        for i, name in enumerate(foot_names):
            fid = foot_ids[i]
            # Get frame placement relative to world (which is base here)
            pos = data.oMf[fid].translation
            heights[name].append(pos[2])

    return times, heights


def compute_contact_status(df: pd.DataFrame, force_threshold: float = 5.0) -> Optional[List[np.ndarray]]:
    """Determine contact boolean status based on force threshold."""
    foot_force_cols = ['foot_force_0', 'foot_force_1', 'foot_force_2', 'foot_force_3']
    
    if not all(col in df.columns for col in foot_force_cols):
        print("Warning: Foot force columns missing. Skipping contact viz.")
        return None
    
    contact_data = []
    for col in foot_force_cols:
        contact = (df[col] > force_threshold).values
        contact_data.append(contact)
    
    return contact_data


def plot_heights(times: List[str], 
                 heights: Dict[str, List[float]], 
                 contact_data: List[np.ndarray] = None, 
                 force_threshold: float = 25):
    """Generate and save the visualization plot."""
    
    # Set a nice style (fallback to standard if unavailable)
    try:
        plt.style.use('bmh') # 'seaborn-whitegrid' is deprecated in newer matplotlib versions
    except:
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                   gridspec_kw={'height_ratios': [1, 2.5]}, 
                                   sharex=True)

    foot_names = list(heights.keys())
    # Define a consistent color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    foot_colors = {name: colors[i % len(colors)] for i, name in enumerate(foot_names)}

    # --- Plot 1: Contact Status ---
    if contact_data:
        # We need to map the contact data indices (0-3) to the feet we are actually plotting
        # Assuming contact_data is always [FL, FR, RL, RR]
        all_feet_order = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        
        for i, name in enumerate(foot_names):
            if name in all_feet_order:
                idx = all_feet_order.index(name)
                color = foot_colors[name]
                
                # Plot bars
                ax1.fill_between(
                    range(len(contact_data[idx])), 
                    i - 0.25, i + 0.25, 
                    where=contact_data[idx], 
                    alpha=0.7, 
                    color=color, 
                    label=f'{name} Contact',
                    linewidth=0
                )

        ax1.set_ylabel('Foot Contact')
        ax1.set_ylim(-0.5, len(foot_names) - 0.5)
        ax1.set_yticks(range(len(foot_names)))
        ax1.set_yticklabels(foot_names)
        ax1.set_title(f'Foot Contact Status (Force Threshold: {force_threshold}N)')
        ax1.grid(True, alpha=0.3, axis='x')
        # Put legend outside if needed, or upper right
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small', framealpha=0.9)

    # --- Plot 2: Heights ---
    for name, data in heights.items():
        ax2.plot(data, label=name, linewidth=2, color=foot_colors[name])

    ax2.set_ylabel('Height relative to Base (m)')
    ax2.set_title('Foot Heights Relative to Base')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.9)

    # --- Axis Formatting ---
    if times:
        start_t, end_t = times[0], times[-1]
        ax2.set_xlabel(f'Time Steps (Start: {start_t}, End: {end_t})')
        # Optional: Add simple X-axis ticks if duration is long
        step_size = max(len(times) // 10, 1)
        ax2.set_xticks(range(0, len(times), step_size))
        # Uncomment below to show actual time strings on X axis (can be crowded)
        # ax2.set_xticklabels([times[i] for i in range(0, len(times), step_size)], rotation=15)
    
    plt.tight_layout()
    
    output_path = PATH_IMGS / 'foot_heights_rel_base.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")
    plt.show()


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Visualize foot heights relative to base from CSV data.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--urdf', type=str, default='resources/go2.urdf', help='Path to the URDF file')
    parser.add_argument('--start', type=str, help='Start time (H:M:S or H:M:S.ff)')
    parser.add_argument('--end', type=str, help='End time (H:M:S or H:M:S.ff)')
    parser.add_argument('--feet', type=str, nargs='+', help='List of feet to plot (e.g., FL FR RL RR)')
    parser.add_argument('--force-threshold', type=float, default=25, help='Force threshold for contact (default: 25)')
    
    args = parser.parse_args()
    
    # 1. Load Model
    model, data = load_model(args.urdf)
    
    # 2. Process Data
    df = process_data(args.csv_file, args.start, args.end)
    
    if df.empty:
        print("No data found for the specified time range.")
        sys.exit(0)

    print(f"Processing {len(df)} rows...")
    print(f"Time range: {df['wall_time'].iloc[0]} -> {df['wall_time'].iloc[-1]}")

    # 3. Compute Heights
    times, heights = compute_foot_heights(model, data, df, args.feet)
    
    # 4. Compute Contact
    contact_data = compute_contact_status(df, force_threshold=args.force_threshold)
    
    # 5. Visualize
    plot_heights(times, heights, contact_data=contact_data, force_threshold=args.force_threshold)


if __name__ == "__main__":
    main()