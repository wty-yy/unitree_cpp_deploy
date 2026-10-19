<div align="center">
  <h1 align="center">G1 RL CPP Deploy</h1>
  <p align="center"><a href="README.md">English</a> | <strong>中文</strong> | <a href="CHANGELOG.md">更新记录</a></p>
</div>

在 Unitree G1 29 自由度人形机器人上部署强化学习与模仿学习策略，可在 Orin NX 上运行，也可通过以太网连接外部 Linux 电脑。支持基础运控、BFM-Zero（`goal`、`reward`、`tracking`）、OmniXtreme 和 BeyondMimic。参考 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)。

## 安装与编译

```bash
# 安装编译依赖和 Git LFS
sudo apt install build-essential cmake git-lfs libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev zlib1g-dev
git lfs install

# 将 G1 分支克隆到 unitree_cpp_deploy_g1 目录
git clone --branch g1 --single-branch https://github.com/wty-yy/unitree_cpp_deploy.git unitree_cpp_deploy_g1
cd unitree_cpp_deploy_g1
git lfs pull
```

安装 [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)，将对应平台的 ONNX Runtime 解压到 `deploy/thirdparty/`。

| 部署平台 | ONNX Runtime | deploy/thirdparty 下的目录 |
|---|---|---|
| Linux ARM64 / Orin NX | [aarch64 GPU 1.24.0](https://drive.google.com/file/d/1y8JJkzSfARwLXMRgpk70DfJFv4gmyNkg/view?usp=sharing) | `onnxruntime-linux-aarch64-gpu-1.24.0` |
| Linux x86_64 | [x64 GPU 1.24.2](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) | `onnxruntime-linux-x64-gpu-1.24.2` |

CMake 检查目标系统，并根据 `CMAKE_SYSTEM_PROCESSOR` 自动选择 `aarch64` / `arm64` 或 `x86_64` / `amd64` 对应的安装包。安装包需包含 `include/onnxruntime_cxx_api.h` 和 `lib/libonnxruntime.so`。`cnpy` 已包含在仓库中，会自动编译，用于读取 NumPy 动作与潜变量文件。

```bash
cd deploy/robots/g1
cmake -B build
cmake --build build -j$(nproc)

# 按需指定其他 ONNX Runtime 安装目录
cmake -B build -DONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime
cmake --build build -j$(nproc)

# 恢复按架构自动选择
cmake -B build -DONNXRUNTIME_ROOT=
```

Orin NX 环境使用 JetPack 6.2 和 CUDA 12.6.68，刷机参考 [G1 刷机记录](https://wty-yy.github.io/posts/30579/#g1%E5%88%B7%E6%9C%BA%E8%AE%B0%E5%BD%95)。需安装所选 GPU 包及执行后端要求的 CUDA/cuDNN/TensorRT 依赖。

### 策略与动作文件

仓库包含当前 [config.yaml](deploy/robots/g1/config/config.yaml) 引用的运控与 BeyondMimic 模型、部署 YAML 和动作 CSV；BFM-Zero 与 OmniXtreme 由用户单独下载。ONNX 和 NPZ 分别通过 `.gitattributes` 中的 `*.onnx`、`*.npz` 规则由 Git LFS 管理，克隆后运行 `git lfs pull` 获取完整文件。

| 状态 | 策略目录 |
|---|---|
| `Velocity_Y` | [logs/g1/g1_rl_lab/mixed/g1_mixed_moe_cts_mixed0.1.5_amp0.05_stadium_control_38k](logs/g1/g1_rl_lab/mixed/g1_mixed_moe_cts_mixed0.1.5_amp0.05_stadium_control_38k) |
| `Velocity_A` | [logs/g1/g1_rl_lab/crawl/g1_crawl_v3.4_8k](logs/g1/g1_rl_lab/crawl/g1_crawl_v3.4_8k) |
| `BFM_goal` | 按 [BFM 下载说明](logs/g1/bfm/.gitkeep) 解压到 `logs/g1/bfm/official/` |
| `OmniXtreme` | 按 [OmniXtreme 下载说明](logs/g1/omnixtreme/.gitkeep) 解压到 `logs/g1/omnixtreme/official/` |
| `BeyondMimic` | [logs/g1/beyondmimic/crawl_v1](logs/g1/beyondmimic/crawl_v1) |

`Velocity_X.policy_dir` 当前为 `null`，启动时禁用该状态。添加其他部署资源时，使用 `git add -f` 将被忽略的 `logs/` 目录下所需模型、部署参数和动作文件一起加入版本控制。

BFM-Zero 和 OmniXtreme 的 `official/` 目录按上述说明单独下载，模型、潜变量与轨迹不随仓库提交。必需文件缺失的状态会在启动时禁用，并在触发跳转时提示原因。

## 运行

### 真机

在 `deploy/robots/g1/` 目录运行，网卡按实际连接修改。启动检查发现 `lowcmd` 已有发布者时，程序会请求官方运控释放控制权，再接管控制；其他自定义运控程序需先关闭。

```bash
./build/g1_ctrl -n eth0
```

### MuJoCo 仿真

先启动使用兼容 Unitree DDS 话题的 G1 29 自由度仿真器，再在另一终端通过环回网卡启动控制程序。

```bash
./build/g1_ctrl -n lo
```

### 按键操作

| 按键 | 功能 |
|---|---|
| LT + Up | Passive → FixStand |
| RB + Y / X / A | 从 FixStand 或其他策略切换到 `Velocity_Y` / `Velocity_X` / `Velocity_A` |
| RT + Y | 切换到 `BFM_goal` |
| RT + X | 切换到 `OmniXtreme` |
| RT + A | 打开 BeyondMimic 动作选择器 |
| Up / Down，然后 A / B | 选择器中：上一条 / 下一条动作，确认 / 取消 |
| LT + B | 从站立或策略状态返回 Passive 阻尼模式 |
| 按住 RT | 当前运控模型配置 boost 范围时，使用加速指令范围 |

`BFM_goal` 中按 `Y` 切换目标，按 `X` 重置；BFM tracking 还支持按 `B` 启动动作播放。OmniXtreme 进入后默认暂停，按 `B` 开始或暂停，按 `Y` / `A` 切换下一条 / 上一条轨迹，按 `X` 重置。状态跳转和按键见 [config.yaml](deploy/robots/g1/config/config.yaml)。

### 命令行参数

| 参数 | 用途 |
|---|---|
| `-n, --network <interface>` | DDS 网卡，如 `eth0` 或 `lo` |
| `-h, --help` | 显示帮助 |
| `-v, --version` | 显示版本 |
| `--log` | 将控制程序输出保存到 `deploy/robots/g1/log/log.txt` |

## 使用

### 仓库结构

```text
deploy/
├── include/         # FSM、观测、推理、滤波与外部速度指令接口
├── robots/g1/       # G1 入口、状态、配置与 CMakeLists.txt
├── tests/           # 外部速度指令接口检查
├── thirdparty/      # ONNX Runtime 安装包、cnpy 与 JSON
└── tools/           # 视频与里程计工具
logs/g1/             # G1 策略、潜变量与动作数据
docs/                # 部署、FSM 配置与观测说明
CHANGELOG.md         # 更新记录
```

主配置文件位于 [deploy/robots/g1/config/config.yaml](deploy/robots/g1/config/config.yaml)。

### 基础状态配置

- `FSM._`: 声明启用的状态及其 `id/type`
- `FixStand`: 配置站立姿态的 `kp`、`kd`、`qs`
- `Velocity_Y` / `Velocity_X` / `Velocity_A`: 通过 `policy_dir` 指定基础运控模型目录
- `transitions`: 使用手柄 DSL 定义状态跳转条件

### 更换策略模型

要更换基础运控模型，修改 `config.yaml` 中对应的 `Velocity_*.policy_dir`：

```yaml
Velocity_Y:
  policy_dir: ../../../logs/g1/g1_rl_lab/mixed/g1_mixed_moe_cts_mixed0.1.5_amp0.05_stadium_control_38k
```

路径相对于 `deploy/robots/g1/` 解析，模型与部署参数需一起替换。对应目录通常需要包含：

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

### BeyondMimic 配置

`FSM.BeyondMimic` 配置 `policy_dir`、`deploy_yaml`（默认 `param/deploy.yaml`）、`onnx_model`、`motion_files`、`fps`、`joint_vel_reference`、`z_axis_yaw_projection`、`time_start` / `time_end` 和 `finished_transition`。动作选择器读取该状态的 `motion_files`，可配置 CSV 文件或目录；当前配置执行结束后转入 `BFM_goal`。

### 日志、加速与工具

- `Velocity_*.logging` 启用策略 CSV 日志，`logging_dt` 设置采样间隔（秒），`logging_options` 选择记录字段；文件保存到 `{policy_dir}/logs/`
- `commands.base_velocity.ranges` 定义普通速度范围；模型配置 `commands.base_velocity.boost.ranges` 后，可通过状态的 `boost_button`（默认 `RT`）触发加速范围
- 保留关节 LPF / Kalman 滤波、固定指令与外部速度指令接口；外部速度接口用于 `Velocity_X` / `Velocity_A`，手柄操作优先
- [FSM 与按键配置](docs/robot_params.md)、[观测分组](docs/obs_group.md)、[BFM / OmniXtreme 部署细节](docs/g1_setup_zh.md)
- [视频抽帧](deploy/tools/video/extract_frames_from_video.py)、[里程计可视化](deploy/tools/visualize_position.py)

### 终止与排查

- 基础 `Velocity_*` 状态默认使用姿态异常检查，可通过状态的 `enable_bad_orientation_check` 配置，阈值当前在 [deploy/robots/g1/src/State_RLBase.cpp](deploy/robots/g1/src/State_RLBase.cpp) 中固定为 `1.0 rad`
- 如果 `OmniXtreme` 推理能跑但动作异常，优先检查 `joint_ids_map`、轨迹首帧姿态以及 `root_body_index / anchor_body_index`
- 如果 BFM 报错 `Observation term 'xxx' is not registered`，优先检查 `deploy.yaml` 中观测项名称是否与 C++ 已注册观测一致

## 许可证

项目采用 [MIT 许可证](LICENSE)。
