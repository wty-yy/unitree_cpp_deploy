<div align="center">
  <h1 align="center">G1 RL CPP Deploy</h1>
  <p align="center">
    <span>English</span> | <a href="README_zh.md">Chinese</a> | <a href="UPDATE.md">Changelog</a>
  </p>
</div>

This project deploys reinforcement learning and motion imitation policies on the Unitree G1 humanoid robot. The current main entry is under `deploy/robots/g1/`. The codebase supports basic locomotion control, BFM-Zero, OmniXtreme, and BeyondMimic motion replay. The repository also still contains the Go2-related implementation under `deploy/robots/go2/`.

## Table of Contents

- [Code Structure](#code-structure)
- [Dependencies](#dependencies)
- [Weight Download](#weight-download)
- [Build Steps](#build-steps)
- [Running Guide](#running-guide)
  - [Basic Usage](#basic-usage)
  - [Operation Flow](#operation-flow)
  - [Runtime Interaction](#runtime-interaction)
- [Configuration](#configuration)
  - [Base State Configuration](#base-state-configuration)
  - [Changing Policy Models](#changing-policy-models)
  - [BFM-Zero Configuration](#bfm-zero-configuration)
  - [OmniXtreme Configuration](#omnixtreme-configuration)
  - [Termination and Troubleshooting](#termination-and-troubleshooting)

## Code Structure

```bash
unitree_cpp_deploy/
├── deploy/                     # Deployment-related code
│   ├── include/                # Common headers (FSM, Isaac Lab interfaces, etc.)
│   ├── robots/g1/              # G1 main program, CMakeLists, and config
│   ├── robots/go2/             # Go2-related implementation
│   └── thirdparty/             # Third-party libraries (onnxruntime, cnpy, json)
├── docs/                       # Supplementary documents
├── logs/                       # Policy models and motion data
├── UPDATE.md                   # Changelog
└── README.md
```

## Dependencies

The real-robot setup currently uses JetPack 6.2 and CUDA 12.6.68. For flashing notes, see [blog - G1 flashing notes](https://wty-yy.github.io/posts/30579/#g1%E5%88%B7%E6%9C%BA%E8%AE%B0%E5%BD%95).

Before building and running, make sure the development environment has the following packages installed:

```bash
sudo apt install libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev zlib1g-dev
```

- **[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)**: Unitree robot development SDK, required for real-robot DDS communication.
- **onnxruntime**:
  - x64 Linux GPU: download [onnxruntime-linux-x64-gpu-1.24.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) and extract it into `deploy/thirdparty/`
  - Orin NX GPU: download [onnxruntime-linux-aarch64-gpu-1.24.0.tar.gz](https://drive.google.com/file/d/1y8JJkzSfARwLXMRgpk70DfJFv4gmyNkg/view?usp=sharing) and extract it into `deploy/thirdparty/`
  - Manual build:
    ```bash
    git clone --recursive https://github.com/microsoft/onnxruntime
    cd onnxruntime
    uv venv  # create a uv virtual environment with a recent CMake for building
    uv pip install cmake
    source ./venv/bin/activate
    ./build.sh --config Release --parallel --build_shared_lib --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
    # Headers are under include/onnxruntime/ and shared libraries are under build/Linux/Release/
    ```
  - After downloading or building, also check the include and link path comments in [deploy/robots/g1/CMakeLists.txt](deploy/robots/g1/CMakeLists.txt).
- **cnpy**: Used to read `npy/npz` files. It is already vendored in the repository, so no extra system package is needed.

## Weight Download

- `Velocity` basic locomotion weights are small and are available under `logs/g1/velocity/` by default
- `BFM-Zero` weights are larger; download [official_bfm.tar.zst](https://drive.google.com/file/d/1cvdXCLbvyO22YmiV5_FiQPpcx9g3vnGM) into `logs/g1/bfm/` and extract it
- `OmniXtreme` weights are larger; download [official_omnixtreme.tar.zst](https://drive.google.com/file/d/1ffYiU07X2I-bpAYFBqg3ekJ4VNndMIrL/view?usp=sharing) into `logs/g1/omnixtreme/` and extract it

```bash
cd logs/g1/bfm
tar -xvf official_bfm.tar.zst
```

## Build Steps

1. Enter the G1 deployment directory:
   ```bash
   cd deploy/robots/g1
   ```

2. Run CMake and build:
   ```bash
   cmake -B build
   cmake --build build -j$(nproc)
   ```

## Running Guide

After building, run the generated `g1_ctrl` executable under `deploy/robots/g1/`.

### Basic Usage

#### Command Line Arguments

- `-h, --help`: show help information
- `-v, --version`: show version information
- `--log`: enable logging and write output to `log/log.txt` under the project root
- `-n, --network <interface>`: specify the DDS network interface, such as `lo` or `eth0`

#### Launch Examples

Loopback test on the local machine:

```bash
./build/g1_ctrl -n lo
```

Real robot example:

```bash
./build/g1_ctrl -n eth0
```

> Before launching, make sure no other process is occupying the `lowcmd` channel, otherwise the program will report a control conflict.

### Operation Flow

1. After startup, the console first prints `Waiting for connection to robot...`
2. After a successful connection, the program prints `Connected to robot.`
3. `Passive -> FixStand`: press `LT + Up`
4. Enter control states from `FixStand`:
   - `RB + Y` -> `Velocity_Y`
   - `RB + X` -> `Velocity_X`
   - `RT + Y` -> `BFM_goal`
   - `RT + X` -> `OmniXtreme`
5. Return from `Velocity_Y / Velocity_X / BFM_goal / OmniXtreme / BeyondMimic` to `Passive`: press `LT + B`
6. In the current default configuration, `Velocity_Y`, `Velocity_X`, `BFM_goal`, and `OmniXtreme` can switch directly between each other without returning to `FixStand` first

### Runtime Interaction

#### BFM-Zero

The code supports all three task types: `goal`, `reward`, and `tracking`. The current default sample configuration is `BFM_goal`. The default interactions are:

- `Y.on_pressed`: switch to the next latent or goal
- `X.on_pressed`: reset the current state
- `B.on_pressed`: start motion playback in `tracking` mode

#### OmniXtreme

- `B.on_pressed`: start or pause execution
- `Y.on_pressed`: switch to the next trajectory
- `A.on_pressed`: switch to the previous trajectory
- `X.on_pressed`: reset the current trajectory

After entering `OmniXtreme`, the controller starts in a paused standing state. Press `B` to start executing the current trajectory.

## Configuration

The main configuration file is [deploy/robots/g1/config/config.yaml](deploy/robots/g1/config/config.yaml).

### Base State Configuration

- `FSM._`: declares enabled states and their `id/type`
- `FixStand`: configures standing posture `kp`, `kd`, and `qs`
- `Velocity_Y` / `Velocity_X`: specify the base locomotion model directory with `policy_dir`
- `transitions`: defines state transitions using the gamepad DSL

### Changing Policy Models

To change the base locomotion model, modify `Velocity_Y.policy_dir` or `Velocity_X.policy_dir` in `config.yaml`:

```yaml
Velocity_Y:
  policy_dir: ../../../logs/g1/velocity/g1_moe_cts_v0.0.5.1
```

The target directory should usually contain:

- `exported/policy.onnx`
- `params/deploy.yaml`

### BFM-Zero Configuration

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

### OmniXtreme Configuration

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

### Termination and Troubleshooting

- Base `Velocity_*` states use an abnormal orientation check by default. The threshold is currently hardcoded as `1.0 rad` in [deploy/robots/g1/src/State_RLBase.cpp](deploy/robots/g1/src/State_RLBase.cpp).
- If `OmniXtreme` can run inference but produces incorrect motions, first check `joint_ids_map`, the first frame posture of the trajectory, and `root_body_index / anchor_body_index`.
- If BFM reports `Observation term 'xxx' is not registered`, first check whether the observation names in `deploy.yaml` match the observation terms registered on the C++ side.
