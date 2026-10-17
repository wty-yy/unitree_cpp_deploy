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

| Platform | ONNX Runtime |
|---|---|
| Orin NX (default) | [aarch64 1.16.0](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2) |
| x64 Linux | [x64 1.24.2](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) |

For x64 deployment, switch both the include and library paths in [CMakeLists.txt](deploy/robots/go2/CMakeLists.txt).

```bash
cmake -S deploy/robots/go2 -B deploy/robots/go2/build
cmake --build deploy/robots/go2/build -j4
```

## Run

For hardware deployment, disable the `mcf/*` services through the mobile app and confirm that the official controller has stopped before starting this controller.

```bash
# Orin NX: select the interface connected to the robot
./deploy/robots/go2/build/go2_ctrl -n eth0
# unitree_mujoco simulation: domain_id: 0, use_joystick: 1
./deploy/robots/go2/build/go2_ctrl -n lo
# Command-line help
./deploy/robots/go2/build/go2_ctrl --help
```

| Button | Action |
|---|---|
| L2 + A | Passive → stand; policy → squat; squat → stand |
| Start + Up / Down / Left / Right | Select a policy from standing or another policy |
| L2 + B | Enter Passive damping mode |
| Hold RT | Use the boost command ranges defined by the current model |
| L2 + Y | Toggle a fixed command after enabling `fixed_command.enabled` |

The squat trajectory automatically finishes in Passive mode. Fixed commands are disabled by default. See [config.yaml](deploy/robots/go2/config/config.yaml) for transitions and state parameters.

## Policies

The five policies live in `logs/go2/`. Each contains `exported/policy.onnx` and `params/deploy.yaml`. ONNX files are managed with Git LFS.

| State | Policy directory (relative to logs/go2) |
|---|---|
| Velocity_Up | `go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_74k_0.687` |
| Velocity_Down | `go2_rl_gym/go2_moe_cts_164k_0.6715` |
| Velocity_Left | `go2_rl_lab/go2_mixed_moe_cts_v1.3_8192_89k` |
| Velocity_Right | `go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_82.5k_0.6929` |
| Unassigned | `go2_rl_robotlab/go2_moe_cts_v4.2_176k_0.6984` |

The `go2_rl_robotlab` policy is included as an extra checkpoint. Set a state's `policy_dir` to its directory to use it.

Normal speed ranges come from `commands.base_velocity.ranges` in each model's `params/deploy.yaml`. Up, Left, and Right also define `boost.ranges`:

| State | Longitudinal (m/s) | Lateral (m/s) | Yaw (rad/s) |
|---|---|---|---|
| Velocity_Up | `[-4.5, 4.2]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` |
| Velocity_Left | `[-4.0, 4.0]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` |
| Velocity_Right | `[-4.5, 3.0]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` |

Hold RT to use boost ranges and release it to restore normal ranges. `boost_button` selects another button; omitting it defaults to RT. Down and the extra `go2_rl_robotlab` policy have no boost configuration. Fixed commands ignore boost.

Policy paths resolve relative to `deploy/robots/go2/`. Replace `policy.onnx` and `deploy.yaml` together when changing a model.

## Configuration and tools

The optional parameters at the end of [config.yaml](deploy/robots/go2/config/config.yaml) belong to `Velocity_Right` and apply only to that state. Copy the settings you need into `FSM.Velocity_Up/Down/Left` to use them with another policy.

| Configuration | Function |
|---|---|
| `velocity_command_damping` | Longitudinal command damping and acceleration/deceleration limits |
| `logging` | `enabled` toggles logging; `dt` sets the sample interval in seconds; CSV files go in the policy directory's `logs/` |
| `fixed_command` | Toggle a preset velocity command with L2 + Y |
| `joint_pos_udp` | Send the 12 measured joint positions over UDP |

See [robot_params.md](docs/robot_params.md) for FSM and button settings, and [obs_group.md](docs/obs_group.md) for observation groups. [udp_test.py](scripts/udp_test.py) provides a UDP receiver example. `deploy/tools/` contains log and joint-position visualization scripts; `resources/go2.urdf` is the Go2 model.

## License

Original project code uses the [MIT license](LICENSE). Apache 2.0 terms for Unitree-derived code are in the [Unitree RL Lab license](docs/licenses/unitree-rl-lab-license.txt).
