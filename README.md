<div align="center">
	<h1 align="center">Go2 RL CPP Deploy</h1>
	<p align="center">
		<span>🌎 English</span> | <a href="README_zh.md">🇨🇳 中文</a>
	</p>
</div>

This project is designed for deploying Reinforcement Learning (RL) policies on the Unitree Go2 Edu robot (equipped with Orin NX). The code is modified based on [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), support deployment via Ethernet or onboard computer.

## Directory Structure

```
unitree_rl_lab/
├── deploy/                 # Deployment related code
│   ├── include/            # Common headers (FSM, Isaac Lab interfaces, etc.)
│   ├── robots/go2/         # Main program, CMakeLists, and config for Go2 robot
│   └── thirdparty/         # Third-party libraries (onnxruntime, json)
├── logs/                   # Directory for trained RL policy models
└── README.md
```

## Dependencies

Before compiling and running, please ensure the following dependencies are installed on your development environment (Orin NX):

- **[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)**: SDK for Unitree robot development.
- **Boost**: For program options parsing (`program_options`).
- **yaml-cpp**: For reading configuration files.
- **Eigen3**: Matrix operation library.
- **fmt**: Formatting library.
- **onnxruntime**:
   - Orin NX: Download [onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2) and extract it to the `deploy/thirdparty/` folder.
   - x64 Linux: Download [onnxruntime-linux-x64-1.23.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz) and extract it to the `deploy/thirdparty/` folder. Modify the ONNX link path in [{ROBOT}/CMakeLists.txt](deploy/robots/go2/CMakeLists.txt) (uncomment the corresponding lines).

> If deploying on the Orin NX onboard computer, install the aarch64 version of onnxruntime; if deploying via a x64 Linux computer connected by Ethernet, install the x64 version of onnxruntime and modify the ONNX link path accordingly.

## Compilation Steps

1. Enter the Go2 deployment directory:
   ```bash
   cd deploy/robots/go2
   ```

2. Create a build directory:
   ```bash
   mkdir build && cd build
   ```

3. Run CMake and compile:
   ```bash
   cmake .. && make -j8
   ```

## Running Guide

After compilation, run the generated `go2_ctrl` executable in the `build` directory.

### Basic Usage

#### Command Line Arguments

- `-h, --help`: Show help message.
- `-v, --version`: Show version information.
- `--log`: Enable logging (logs will be saved in `deploy/robots/go2/log/`).
- `-n, --network <interface>`: Specify the network interface name for DDS communication (e.g., `eth0`, `wlan0`). If not specified, the default interface will be used.

#### Real Robot Launch
Start the Go2 robot. After it enters the standing state, press **[L2+A]** twice to make the robot lie down. Connect via the mobile App, go to Settings -> Service Status, turn off `mcf/*` services and close the official control program to avoid control conflicts.

```bash
sudo ./go2_ctrl [options]
```

**Example:**

Start on NX, assuming the network interface is `eth0`:
```bash
./go2_ctrl -n eth0
```

#### Simulation Launch

You can control the robot using a gamepad with the Xbox data protocol. If you need other protocols, please modify the joystick configuration file in `unitree_mujoco`.

Download and compile the contents of `simulate/` in [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco), and configure `simulate/config.yaml` with `domain_id: 0`, `use_joystick: 1`.
```bash
./simulate/build/unitree_mujoco  # Start simulation
./go2_ctrl -n lo  # Start control
```

### Operation Flow

1. After starting the program, the console will display "Waiting for connection to robot..."
2. Ensure the robot is powered on and connected. Upon successful connection, it will display "Connected to robot."
3. **Enter Stand Mode**: Press **[L2 + A]** on the gamepad. The robot will enter `FixStand` mode and stand up.
4. **Start RL Control**: Press **[Start + Up/Down/Left/Right]** on the gamepad. The robot will switch to the corresponding `policy_dir_[up,down,left,right]` control model and begin executing the RL policy.
5. **Switch Model**: During operation, you can switch to different RL models at any time by pressing **[Start + Direction Key]**.
6. **Fixed Command Execution**: Press **[L2 + Y]** on the gamepad. The robot will start executing preset fixed commands (as set in the config file). Press the combination again to stop fixed command execution.
7. **Enter Damping Mode**: Press **[L2 + B]** on the gamepad. The robot will enter damping mode and stop RL control.

https://github.com/user-attachments/assets/56375e44-5ac1-42b0-a268-7837c2857287

## Configuration

The configuration file is located at [deploy/robots/go2/config/config.yaml](./deploy/robots/go2/config/config.yaml). Features include:
1. **Multi-model selection**: Configure `Velocity/policy_dir_up/down/left/right` to specify paths for four models. Switch models using `Start + Direction Key`.
2. **Real-time data logging**: Configure `Velocity/logging: true` and set recording frequency `logging_dt` (default 100Hz). Logs are stored in the model folder, e.g., `./logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745/logs/`.
3. **Fixed command execution**: Configure `Velocity/fixed_command/enabled: true`, set fixed command values `command`, and duration `duration` (optional). Press `L2 + Y` to start/stop.

### Changing Policy Models

To change the RL policy used, modify the `Velocity.policy_dir` fields in `config.yaml`:

```yaml
Velocity:
  # Policy model path (relative to project root or absolute path)
  policy_dir_up: ../../../logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745  # best model
  policy_dir_down: ../../../logs/rsl_rl/go2_moe_cts_fast_flat_v4_fz_54k  # fast model
  policy_dir_left: ../../../logs/rsl_rl/kaiwu2025_v6-2_124004  # not bad model
  policy_dir_right: null
```

The specified directory structure should contain:
- `exported/policy.onnx`: Exported ONNX policy model.
- `params/deploy.yaml`: Corresponding deployment parameters.

### Modifying Control Parameters

You can also adjust PD parameters (`kp`, `kd`) and target joint angles (`qs`) for `FixStand` mode in `config.yaml`.

### Modifying Unexpected Termination Parameters

The control program terminates if the angle between the base_link Z-axis and gravity exceeds a threshold (default 2 rad).

Find `bad_orientation` in [State_RLBase.cpp](deploy/robots/go2/src/State_RLBase.cpp) and modify the second parameter to change the radian threshold.
