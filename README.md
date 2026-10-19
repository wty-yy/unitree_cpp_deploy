<div align="center">
  <h1 align="center">G1 RL CPP Deploy</h1>
  <p align="center"><strong>English</strong> | <a href="README.zh-CN.md">中文</a> | <a href="CHANGELOG.md">Changelog</a></p>
</div>

Deploy reinforcement learning and motion imitation policies on the Unitree G1 29-DOF humanoid, using an Orin NX or an external Linux computer connected over Ethernet. Supports locomotion, BFM-Zero (`goal`, `reward`, `tracking`), OmniXtreme, and BeyondMimic. Developed with reference to [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

## Setup and build

```bash
# Install build dependencies and Git LFS
sudo apt install build-essential cmake git-lfs libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev zlib1g-dev
git lfs install

# Clone the G1 branch into unitree_cpp_deploy_g1
git clone --branch g1 --single-branch https://github.com/wty-yy/unitree_cpp_deploy.git unitree_cpp_deploy_g1
cd unitree_cpp_deploy_g1
git lfs pull
```

Install [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2), then extract the ONNX Runtime package for your platform into `deploy/thirdparty/`.

| Platform | ONNX Runtime | Package directory under deploy/thirdparty |
|---|---|---|
| Linux ARM64 / Orin NX | [aarch64 GPU 1.24.0](https://drive.google.com/file/d/1y8JJkzSfARwLXMRgpk70DfJFv4gmyNkg/view?usp=sharing) | `onnxruntime-linux-aarch64-gpu-1.24.0` |
| Linux x86_64 | [x64 GPU 1.24.2](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) | `onnxruntime-linux-x64-gpu-1.24.2` |

CMake checks the target system and automatically selects the package using `CMAKE_SYSTEM_PROCESSOR`: `aarch64` / `arm64` or `x86_64` / `amd64`. Each package must contain `include/onnxruntime_cxx_api.h` and `lib/libonnxruntime.so`. `cnpy` is vendored and built automatically for NumPy motion and latent files.

```bash
cd deploy/robots/g1
cmake -B build
cmake --build build -j$(nproc)

# Use another ONNX Runtime installation if needed
cmake -B build -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime
cmake --build build -j$(nproc)

# Restore architecture-based selection
cmake -B build -DONNXRUNTIME_ROOT=
```

The Orin NX setup uses JetPack 6.2 and CUDA 12.6.68; see [G1 flashing notes](https://wty-yy.github.io/posts/30579/#g1%E5%88%B7%E6%9C%BA%E8%AE%B0%E5%BD%95). Install the CUDA/cuDNN/TensorRT dependencies required by the selected GPU package and execution provider.

### Policy and motion files

The repository includes the locomotion and BeyondMimic models, deployment YAML, and motion CSV referenced by the current [config.yaml](deploy/robots/g1/config/config.yaml). Download BFM-Zero and OmniXtreme separately. ONNX and NPZ files use Git LFS through the `*.onnx` and `*.npz` rules in `.gitattributes`. Run `git lfs pull` after cloning to download the full files.

| State | Policy directory |
|---|---|
| `Velocity_Y` | [logs/g1/g1_rl_lab/mixed/g1_mixed_moe_cts_mixed0.1.5_amp0.05_stadium_control_38k](logs/g1/g1_rl_lab/mixed/g1_mixed_moe_cts_mixed0.1.5_amp0.05_stadium_control_38k) |
| `Velocity_A` | [logs/g1/g1_rl_lab/crawl/g1_crawl_v3.4_8k](logs/g1/g1_rl_lab/crawl/g1_crawl_v3.4_8k) |
| `BFM_goal` | Follow the [BFM download instructions](logs/g1/bfm/.gitkeep) and extract to `logs/g1/bfm/official/` |
| `OmniXtreme` | Follow the [OmniXtreme download instructions](logs/g1/omnixtreme/.gitkeep) and extract to `logs/g1/omnixtreme/official/` |
| `BeyondMimic` | [logs/g1/beyondmimic/crawl_v1](logs/g1/beyondmimic/crawl_v1) |

`Velocity_X.policy_dir` is currently `null`, so that state is disabled at startup. To add another deployment resource under the ignored `logs/` directory, use `git add -f` for its required models, deployment parameters, and motion files together.

Download the BFM-Zero and OmniXtreme `official/` directories using the instructions above; their models, latent files, and trajectories are not bundled in the repository. States with missing required files are disabled at startup, and their transitions report the reason.

## Run

### Real hardware

Run from `deploy/robots/g1/` and select the interface connected to the robot. When the startup check detects an active `lowcmd` publisher, the program requests release of the official motion controller before taking control. Close any other custom controller first.

```bash
./build/g1_ctrl -n eth0
```

### MuJoCo simulator

Start a G1 29-DOF simulator with compatible Unitree DDS topics, then launch the controller in a separate terminal using the loopback interface.

```bash
./build/g1_ctrl -n lo
```

### Button controls

| Button | Action |
|---|---|
| LT + Up | Passive → FixStand |
| RB + Y / X / A | Select `Velocity_Y` / `Velocity_X` / `Velocity_A` from FixStand or another policy |
| RT + Y | Enter `BFM_goal` |
| RT + X | Enter `OmniXtreme` |
| RT + A | Open the BeyondMimic motion selector |
| Up / Down, then A / B | In the selector: previous / next motion, confirm / cancel |
| LT + B | Return from standing or a policy to Passive damping mode |
| Hold RT | Use the current locomotion model's boost ranges, when configured |

In `BFM_goal`, press `Y` for the next goal and `X` to reset. BFM tracking additionally supports `B` to start motion playback. OmniXtreme enters paused: `B` starts or pauses execution, `Y` / `A` selects the next / previous trajectory, and `X` resets it. Policy transitions and buttons are configured in [config.yaml](deploy/robots/g1/config/config.yaml).

### Command-line options

| Option | Usage |
|---|---|
| `-n, --network <interface>` | DDS network interface, such as `eth0` or `lo` |
| `-h, --help` | Show help |
| `-v, --version` | Show version |
| `--log` | Write controller output to `deploy/robots/g1/log/log.txt` |

## Usage

### Repository layout

```text
deploy/
├── include/         # FSM, observations, inference, filters, external velocity bridge
├── robots/g1/       # G1 entry point, states, configuration, and CMakeLists.txt
├── tests/           # External velocity bridge checks
├── thirdparty/      # ONNX Runtime packages, cnpy, and JSON
└── tools/           # Video and odometry tools
logs/g1/             # G1 policies, latent files, and motion data
docs/                # Setup, FSM configuration, and observation documentation
CHANGELOG.md         # Version history
```

The main configuration file is [deploy/robots/g1/config/config.yaml](deploy/robots/g1/config/config.yaml).

### Base state configuration

- `FSM._`: declares enabled states and their `id/type`
- `FixStand`: configures standing posture `kp`, `kd`, and `qs`
- `Velocity_Y` / `Velocity_X` / `Velocity_A`: specify the base locomotion model directory with `policy_dir`
- `transitions`: defines state transitions using the gamepad DSL

### Changing policy models

To change the base locomotion model, modify the corresponding `Velocity_*.policy_dir` in `config.yaml`:

```yaml
Velocity_Y:
  policy_dir: ../../../logs/g1/g1_rl_lab/mixed/g1_mixed_moe_cts_mixed0.1.5_amp0.05_stadium_control_38k
```

Paths resolve relative to `deploy/robots/g1/`. Replace the model and deployment parameters together. The target directory should contain:

- `exported/policy.onnx`
- `params/deploy.yaml`

### BFM-Zero configuration

Under `FSM.BFM_goal / FSM.BFM_reward / FSM.BFM_tracking`, you can configure:

- `policy_dir`: BFM model directory
- `deploy_yaml`: deployment parameter path, default `param/deploy.yaml`
- `onnx_model`: ONNX model path, default `exported/FBcprAuxModel.onnx`
- `onnx_cuda` / `onnx_tensorrt` / `onnx_cuda_device`: ONNX Runtime backend settings
- `task_type`: `goal`, `reward`, or `tracking`
- `latent_file`: latent `.npz` file for the corresponding task
- `gamepad_map`: overrides for `start_motion`, `next_latent`, and `reset_state`
- `goal.selected_goals`, `reward.selected_rewards_filter_z`, `tracking.*`: task-specific settings

Example BFM model directory layout:

- `exported/FBcprAuxModel.onnx`
- `param/deploy.yaml`
- `goal_inference/goal_reaching.npz`
- `reward_inference/reward_locomotion.npz`
- `tracking_inference/zs_walking.npz`

For BFM, the `deploy.yaml` observations are recommended to use a two-group layout:

- `observations.obs_base`
- `observations.obs_hist`

This avoids duplicate YAML key conflicts and matches the observation concatenation order used by the current `State_BFM` implementation.

### OmniXtreme configuration

Under `FSM.OmniXtreme`, you can configure:

- `policy_dir`
- `deploy_yaml`
- `base_model`
- `residual_model`
- `fk_model`
- `motion_files`
- `onnx_cuda` / `onnx_tensorrt` / `onnx_cuda_device`
- `residual_scale`
- `loop_trajectory`
- `root_body_index`
- `anchor_body_index`
- `gamepad_map.next_trajectory / previous_trajectory / reset_trajectory / toggle_execute`

Example OmniXtreme model directory layout:

- `exported/base_policy_trt.onnx`
- `exported/residual_policy.onnx`
- `exported/fk_trt.onnx`
- `exported/motions/*.npz`
- `params/deploy.yaml`

Control-related constants live under the `omnixtreme` section in `params/deploy.yaml`. At minimum, the following fields must be configured correctly:

- `pd_bias_joint_pos`
- `action_scale`
- `p_gains / d_gains`
- `envelope_x1 / envelope_x2 / envelope_y1 / envelope_y2`
- `friction_va / friction_fs / friction_fd`

### BeyondMimic configuration

`FSM.BeyondMimic` configures `policy_dir`, `deploy_yaml` (default `param/deploy.yaml`), `onnx_model`, `motion_files`, `fps`, `joint_vel_reference`, `z_axis_yaw_projection`, `time_start` / `time_end`, and `finished_transition`. The motion selector reads this state's `motion_files`, which accepts CSV files or directories. The current configuration returns to `BFM_goal` when playback finishes.

### Logging, boost, and tools

- `Velocity_*.logging` enables policy CSV logs; `logging_dt` sets the sampling interval in seconds, and `logging_options` selects fields. Files are saved under `{policy_dir}/logs/`
- `commands.base_velocity.ranges` defines normal speed ranges; when the model defines `commands.base_velocity.boost.ranges`, the state's `boost_button` (default `RT`) activates boost ranges
- Joint LPF / Kalman filters, fixed commands, and the external velocity bridge remain available. The bridge controls `Velocity_X` / `Velocity_A`, with manual controller input taking priority
- [FSM and buttons](docs/robot_params.md), [observation groups](docs/obs_group.md), and [BFM / OmniXtreme setup details](docs/g1_setup_zh.md)
- [Video frame extraction](deploy/tools/video/extract_frames_from_video.py) and [odometry visualization](deploy/tools/visualize_position.py)

### Termination and troubleshooting

- Base `Velocity_*` states use an abnormal orientation check by default, configurable with the state's `enable_bad_orientation_check`. The threshold is currently hardcoded as `1.0 rad` in [deploy/robots/g1/src/State_RLBase.cpp](deploy/robots/g1/src/State_RLBase.cpp).
- If `OmniXtreme` can run inference but produces incorrect motions, first check `joint_ids_map`, the first frame posture of the trajectory, and `root_body_index / anchor_body_index`.
- If BFM reports `Observation term 'xxx' is not registered`, first check whether the observation names in `deploy.yaml` match the observation terms registered on the C++ side.

## License

The project uses the [MIT license](LICENSE).
