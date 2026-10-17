<div align="center">
  <h1>Go2 RL CPP Deploy</h1>
  <p><strong>English</strong> | <a href="README_zh.md">中文</a></p>
</div>

Deploy reinforcement learning policies on Unitree Go2 Edu, using an Orin NX or an external computer connected over Ethernet. Based on [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

## Setup and build

```bash
# Install dependencies and Git LFS
sudo apt install git-lfs libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev
git lfs install

# Clone the go2 branch and download its five ONNX policies
git clone --branch go2 --single-branch https://github.com/wty-yy/unitree_cpp_deploy.git
cd unitree_cpp_deploy
git lfs pull
```

Install [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) and extract the ONNX Runtime package for your architecture into `deploy/thirdparty/`.

| Platform | ONNX Runtime | Path |
|---|---|---|
| Orin NX | [aarch64 1.16.0](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2) | deploy/thirdparty/onnxruntime-linux-aarch64-gpu-1.16.0 |
| x64 Linux | [x64 1.24.2](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) | deploy/thirdparty/onnxruntime-linux-x64-gpu-1.24.2 |

CMake automatically selects the ONNX Runtime package for your architecture.

Build with the following commands:

```bash
cd deploy/robots/go2
cmake -B build && cmake --build build -j$(nproc)
```

## Run

### Real Hardware

For hardware deployment, use the official remote controller to make the robot lie down by pressing L2 + A twice. Then start this motion control program; it automatically stops the official motion controller and takes over control.

```bash
# Orin NX: select the interface connected to the robot
./build/go2_ctrl -n eth0
```

### MuJoco Simulator

For simulation deployment, follow the installation instructions in [unitree_mujoco (go2 branch)](https://github.com/wty-yy/unitree_mujoco/tree/go2). After installation, start the simulator, then start the controller in a separate terminal:

```bash
# From the unitree_mujoco repository root
./simulate/build/unitree_mujoco

# From this repository's deploy/robots/go2 directory in a separate terminal
./build/go2_ctrl -n lo
```

### Button Controls

| Button | Action |
|---|---|
| L2 + A | Passive → stand; stand / policy → squat; squat → stand |
| Start + Up / Down / Left / Right | Select a policy from standing or another policy |
| L2 + B | Enter Passive damping mode |
| Hold RT | Use the boost command ranges defined by the current model |
| L2 + Y | Toggle a fixed command after enabling `fixed_command.enabled` |

The squat trajectory automatically finishes in Passive mode. Fixed commands are disabled by default. See [config.yaml](deploy/robots/go2/config/config.yaml) for transitions and state parameters.

![Go2 button controls and state transitions](asset/figures/go2_fsm.svg)

Button control demo:

https://github.com/user-attachments/assets/d27f4d4b-208e-4212-b829-ad835e53c162

## Usage

### Policies

The five policies live in [logs/go2/](logs/go2/). Each contains `exported/policy.onnx` and `params/deploy.yaml`. ONNX files are managed with Git LFS.

| State | Policy directory (relative to logs/go2) |
|---|---|
| Velocity_Up | [go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_74k_0.687](logs/go2/go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_74k_0.687) |
| Velocity_Down | [go2_rl_gym/go2_moe_cts_164k_0.6715](logs/go2/go2_rl_gym/go2_moe_cts_164k_0.6715) |
| Velocity_Left | [go2_rl_lab/go2_mixed_moe_cts_v1.3_8192_89k_0.69](logs/go2/go2_rl_lab/go2_mixed_moe_cts_v1.3_8192_89k_0.69) |
| Velocity_Right | [go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_82.5k_0.6929](logs/go2/go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_82.5k_0.6929) |
| Unassigned | [go2_rl_robotlab/go2_moe_cts_v4.2_176k_0.6984](logs/go2/go2_rl_robotlab/go2_moe_cts_v4.2_176k_0.6984) |

#### Boost Speed

Normal speed ranges come from `commands.base_velocity.ranges` in each model's `params/deploy.yaml`. Up, Left, and Right also define `boost.ranges`:

| State | Longitudinal (m/s) | Lateral (m/s) | Yaw (rad/s) | YAML path |
|---|---|---|---|---|
| Velocity_Up | `[-4.5, 4.2]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` | [deploy.yaml](logs/go2/go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_74k_0.687/params/deploy.yaml#L14) |
| Velocity_Left | `[-4.0, 4.0]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` | [deploy.yaml](logs/go2/go2_rl_lab/go2_mixed_moe_cts_v1.3_8192_89k_0.69/params/deploy.yaml#L14) |
| Velocity_Right | `[-4.5, 3.0]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` | [deploy.yaml](logs/go2/go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_82.5k_0.6929/params/deploy.yaml#L14) |

Boost uses `RT` by default. Configure the button with `boost_button` in [config.yaml](deploy/robots/go2/config/config.yaml). Currently, only `go2_rl_lab` models have boost configured.

Policy paths resolve relative to [deploy/robots/go2/](deploy/robots/go2/). Replace `policy.onnx` and `deploy.yaml` together when changing a model.

### Configuration and tools

[config.yaml](deploy/robots/go2/config/config.yaml) provides the following optional settings under `Velocity_Right`. Set `enabled: true` in each section to enable it; copy the section to another FSM state as needed.

| Configuration | Usage |
|---|---|
| `velocity_command_damping` | Smooth longitudinal commands with `vx.speed_points` and `vx.time_constants`; set `vx.max_acceleration` and `vx.max_deceleration` to limit acceleration and deceleration. |
| `logging` | Set `dt` to the sampling interval in seconds. CSV files are saved in the policy directory's `logs/` folder. |
| `fixed_command` | Set `lin_vel_x`, `lin_vel_y`, `ang_vel_z`, and `duration` (seconds), then press `L2 + Y` to start or stop the preset command. |
| `joint_pos_udp` | Set the receiver's `host` and `port`, and the sending interval `dt` (seconds), to stream the 12 measured joint positions. See [udp_test.py](scripts/udp_test.py) for a receiver example. |

See [robot_params.md](docs/robot_params.md) for FSM and button settings, and [obs_group.md](docs/obs_group.md) for observations. Additional resources: [visualization scripts](deploy/tools/) and [Go2 URDF](resources/go2.urdf).

## License

Original project code uses the [MIT license](LICENSE). Apache 2.0 terms for Unitree-derived code are in the [Unitree RL Lab license](docs/licenses/unitree-rl-lab-license.txt).
