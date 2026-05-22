<div align="center">
  <h1 align="center">G1 RL CPP Deploy</h1>
  <p align="center">
    <span>中文</span> | <a href="README.md">English</a> | <a href="UPDATE.md">更新记录</a>
  </p>
</div>

本项目用于在 Unitree G1 人形机器人上部署强化学习与模仿学习策略，当前主入口位于 `deploy/robots/g1/`。代码支持基础运控、BFM-Zero、OmniXtreme 以及 BeyondMimic 动作复现；仓库中仍保留 `deploy/robots/go2/` 的 Go2 相关实现。

## 目录

- [代码结构](#代码结构)
- [依赖项](#依赖项)
- [权重下载](#权重下载)
- [编译步骤](#编译步骤)
- [运行指南](#运行指南)
  - [基本用法](#基本用法)
  - [操作流程](#操作流程)
  - [运行中交互](#运行中交互)
- [配置说明](#配置说明)
  - [基础状态配置](#基础状态配置)
  - [更换策略模型](#更换策略模型)
  - [BFM-Zero 配置](#bfm-zero-配置)
  - [OmniXtreme 配置](#omnixtreme-配置)
  - [终止与排查](#终止与排查)

## 代码结构

```bash
unitree_cpp_deploy/
├── deploy/                     # 部署相关代码
│   ├── include/                # 通用头文件 (FSM、Isaac Lab 接口等)
│   ├── robots/g1/              # G1 主程序、CMakeLists 和配置
│   ├── robots/go2/             # Go2 相关实现
│   └── thirdparty/             # 第三方库 (onnxruntime、cnpy、json)
├── docs/                       # 补充文档
├── logs/                       # 策略模型与动作数据
├── UPDATE.md                   # 更新记录
└── README.md
```

## 依赖项

真机代码使用的版本为Jetpack 6.2，CUDA 12.6.68，刷机请参考 [blog - G1刷机记录](https://wty-yy.github.io/posts/30579/#g1%E5%88%B7%E6%9C%BA%E8%AE%B0%E5%BD%95)

在编译和运行之前，请确保开发环境已安装以下依赖项：

```bash
sudo apt install libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev zlib1g-dev
```

- **[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)**: Unitree 机器人开发 SDK，真机 DDS 通信依赖。
- **onnxruntime**:
  - x64 Linux GPU: 下载 [onnxruntime-linux-x64-gpu-1.24.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) 解压到 `deploy/thirdparty/`
  - Orin NX GPU: 下载 [onnxruntime-linux-aarch64-gpu-1.24.0.tar.gz](https://drive.google.com/file/d/1y8JJkzSfARwLXMRgpk70DfJFv4gmyNkg/view?usp=sharing) 解压到 `deploy/thirdparty/`
  - 手动编译：
    ```bash
    git clone --recursive https://github.com/microsoft/onnxruntime
    cd onnxruntime
    git checkout v1.24.4
    uv venv  # uv虚拟环境中装上最新的cmake用于编译
    uv pip install cmake
    source ./venv/bin/activate
    ./build.sh --config Release --parallel --build_shared_lib --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
    # 头文件在源码的 include/onnxruntime/ 里，.so 文件则会在 build/Linux/Release/
    ```
  - 下载/编译完成后：请同步检查 [deploy/robots/g1/CMakeLists.txt](deploy/robots/g1/CMakeLists.txt) 中对应的 include 和 link 路径注释。
- **cnpy**: 用于读取 `npy/npz` 文件，已作为子模块接入，无需单独安装系统包。


## 权重下载

- `Velocity` 简单运控权重较小，默认位于 `logs/g1/velocity/`
- `BFM-Zero` 权重较大，请下载 [official_bfm.tar.zst](https://drive.google.com/file/d/1cvdXCLbvyO22YmiV5_FiQPpcx9g3vnGM) 到 `logs/g1/bfm/` 并解压
- `OmniXtreme` 权重较大，请下载 [official_omnixtreme.tar.zst](https://drive.google.com/file/d/1ffYiU07X2I-bpAYFBqg3ekJ4VNndMIrL/view?usp=sharing) 到 `logs/g1/omnixtreme/` 并解压

```bash
cd logs/g1/bfm
tar -xvf official_bfm.tar.zst
```

## 编译步骤

1. 进入 G1 部署目录：
   ```bash
   cd deploy/robots/g1
   ```

2. 运行 CMake 并编译：
   ```bash
   cmake -B build
   cmake --build build -j$(nproc)
   ```

## 运行指南

编译完成后，在 `deploy/robots/g1/` 目录下运行生成的 `g1_ctrl` 可执行文件。

### 基本用法

#### 命令行参数

- `-h, --help`: 显示帮助信息
- `-v, --version`: 显示版本信息
- `--log`: 开启日志记录，输出到项目根目录 `log/log.txt`
- `-n, --network <interface>`: 指定 DDS 通信网卡，例如 `lo`、`eth0`

#### 启动示例

本机环回测试：

```bash
./build/g1_ctrl -n lo
```

真机运行示例：

```bash
./build/g1_ctrl -n eth0
```

> 启动前请确保没有其他进程占用 `lowcmd` 通道，否则程序会提示控制冲突。

### 操作流程

1. 启动程序后，控制台会先显示 `Waiting for connection to robot...`
2. 连接成功后，程序会显示 `Connected to robot.`
3. `Passive -> FixStand`：按 `LT + Up`
4. 从 `FixStand` 进入各控制状态：
   - `RB + Y` -> `Velocity_Y`
   - `RB + X` -> `Velocity_X`
   - `RT + Y` -> `BFM_goal`
   - `RT + X` -> `OmniXtreme`
5. 从 `Velocity_Y / Velocity_X / BFM_goal / OmniXtreme / BeyondMimic` 返回 `Passive`：按 `LT + B`
6. 当前默认配置支持 `Velocity_Y`、`Velocity_X`、`BFM_goal`、`OmniXtreme` 之间直接切换，无需先回 `FixStand`

### 运行中交互

#### BFM-Zero

代码支持 `goal / reward / tracking` 三种任务类型，当前默认示例配置为 `BFM_goal`。默认交互如下：

- `Y.on_pressed`: 切换下一个 latent / goal
- `X.on_pressed`: 重置当前状态
- `B.on_pressed`: 在 `tracking` 任务中启动动作播放

#### OmniXtreme

- `B.on_pressed`: 开始或暂停动作执行
- `Y.on_pressed`: 切换到下一条轨迹
- `A.on_pressed`: 切换到上一条轨迹
- `X.on_pressed`: 重置当前轨迹

进入 `OmniXtreme` 后默认处于暂停站立状态，按 `B` 后开始执行当前轨迹。

## 配置说明

主配置文件位于 [deploy/robots/g1/config/config.yaml](deploy/robots/g1/config/config.yaml)。

### 基础状态配置

- `FSM._`: 声明启用的状态及其 `id/type`
- `FixStand`: 配置站立姿态的 `kp`、`kd`、`qs`
- `Velocity_Y` / `Velocity_X`: 通过 `policy_dir` 指定基础运控模型目录
- `transitions`: 使用手柄 DSL 定义状态跳转条件

### 更换策略模型

要更换基础运控模型，修改 `config.yaml` 中的 `Velocity_Y.policy_dir` 或 `Velocity_X.policy_dir`：

```yaml
Velocity_Y:
  policy_dir: ../../../logs/g1/velocity/g1_moe_cts_v0.0.5.1
```

对应目录通常需要包含：

- `exported/policy.onnx`
- `params/deploy.yaml`

### BFM-Zero 配置

在 `FSM.BFM_goal / FSM.BFM_reward / FSM.BFM_tracking` 中可配置：

- `policy_dir`: BFM 模型目录
- `deploy_yaml`: 部署参数路径，默认 `param/deploy.yaml`
- `onnx_model`: ONNX 模型路径，默认 `exported/FBcprAuxModel.onnx`
- `onnx_cuda` / `onnx_tensorrt` / `onnx_cuda_device`: ONNX Runtime 后端设置
- `task_type`: `goal` / `reward` / `tracking`
- `latent_file`: 对应任务的 `.npz` 潜变量文件
- `gamepad_map`: 可覆盖 `start_motion`、`next_latent`、`reset_state`
- `goal.selected_goals`、`reward.selected_rewards_filter_z`、`tracking.*`: 任务特定参数

BFM 模型目录约定示例：

- `exported/FBcprAuxModel.onnx`
- `param/deploy.yaml`
- `goal_inference/goal_reaching.npz`
- `reward_inference/reward_locomotion.npz`
- `tracking_inference/zs_walking.npz`

BFM 的 `deploy.yaml` 观测建议使用双组结构：

- `observations.obs_base`
- `observations.obs_hist`

这样可以避免 YAML 同名 key 冲突，并与当前 `State_BFM` 的观测拼接顺序保持一致。

### OmniXtreme 配置

在 `FSM.OmniXtreme` 中可配置：

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

OmniXtreme 模型目录约定示例：

- `exported/base_policy_trt.onnx`
- `exported/residual_policy.onnx`
- `exported/fk_trt.onnx`
- `exported/motions/*.npz`
- `params/deploy.yaml`

控制相关常量位于 `params/deploy.yaml` 的 `omnixtreme` 节中，至少需要正确配置：

- `pd_bias_joint_pos`
- `action_scale`
- `p_gains / d_gains`
- `envelope_x1 / envelope_x2 / envelope_y1 / envelope_y2`
- `friction_va / friction_fs / friction_fd`

### 终止与排查

- 基础 `Velocity_*` 状态默认使用姿态异常检查，阈值当前在 [deploy/robots/g1/src/State_RLBase.cpp](deploy/robots/g1/src/State_RLBase.cpp) 中固定为 `1.0 rad`
- 如果 `OmniXtreme` 推理能跑但动作异常，优先检查 `joint_ids_map`、轨迹首帧姿态以及 `root_body_index / anchor_body_index`
- 如果 BFM 报错 `Observation term 'xxx' is not registered`，优先检查 `deploy.yaml` 中观测项名称是否与 C++ 已注册观测一致
