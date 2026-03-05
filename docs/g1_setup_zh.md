# G1人形机器人部署
## 简介

G1推理代码位于`deploy/robots/g1/`目录下，目前支持Loco，BFM-Zero (reward, tracking, goal)的部署

```bash
unitree_rl_lab/
├── deploy/                 # 部署相关的代码
│   ├── include/            # 通用头文件 (FSM, Isaac Lab 接口等)
│   ├── robots/g1/         # G1 机器人的主程序、CMakeLists 和配置
│   └── thirdparty/         # 第三方库 (onnxruntime, json)
└── logs/                   # 存放训练好的 RL 策略模型
```

## 依赖项

```bash
sudo apt install libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev zlib1g-dev
```

BFM-Zero部署还需要安装`cnpy`库读取`npy`潜向量数据，并推荐使用支持CUDA加速的ONNX Runtime库

- **onnxruntime**:
   - x64 Linux GPU: 下载[ onnxruntime-linux-x64-gpu-1.24.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz)解压到`deploy/thirdparty/`文件夹中
   - Orin NX GPU: 下载[onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2)解压到`deploy/thirdparty/`文件夹中，修改[{ROBOT}/CMakeLists.txt](deploy/robots/go2/CMakeLists.txt)中的onnx链接路径(解开相应注释)
- **cnpy**: 读取`npy, npz`文件的C++库，已添加为子模块并包含在CMake中，无需额外安装

## 权重下载

velocity简单无感知运控权重较小，随仓库下载于`logs/g1/velocity`

BFM-Zero模型较大，下载[Google Drive - official_bfm.tar.zst](https://drive.google.com/file/d/1cvdXCLbvyO22YmiV5_FiQPpcx9g3vnGM)到`logs/g1/bfm`文件夹下，解压

```bash
cd logs/g1/bfm
tar -xvf official_bfm.tar.zst
```

## 编译

```bash
cd deploy/robots/g1
cmake -B build && cmake --build build -j$(nproc)
```

## 使用方法
1. 启动控制程序（以本机环回网卡为例）
```bash
./build/g1_ctrl -n lo
```
2. 状态切换（默认按 [config.yaml](deploy/robots/g1/config/config.yaml)）
    - `Passive -> FixStand`: `LT + Up.on_pressed`
    - `FixStand -> Velocity`: `RB + X.on_pressed`
    - `FixStand -> BFM_goal`: `RT + Y.on_pressed`
    - `FixStand -> BFM_reward`: `RT + A.on_pressed`
    - `FixStand -> BFM_tracking`: `RT + X.on_pressed`
    - `Velocity / BFM_* -> Passive`: `LT + B.on_pressed`
    - `Velocity <-> BFM_*` 与 `BFM_* <-> BFM_*` 切换触发与上面保持一致（无需回 FixStand）
3. BFM 运行中手柄交互（默认）
    - `start_motion`: `B.on_pressed`
    - `next_latent`: `Y.on_pressed`
    - `reset_state`: `X.on_pressed`
    - 进入 BFM 状态后策略默认启用

## BFM配置说明
### 1. 目录与文件约定
- 策略目录示例：`logs/g1/bfm/official/`
- ONNX 模型：`exported/FBcprAuxModel.onnx`
- 部署参数：`param/deploy.yaml`
- 潜变量文件：
    - `goal_inference/goal_reaching.npz`
    - `reward_inference/reward_locomotion.npz`
    - `tracking_inference/zs_walking.npz`

### 2. config.yaml 中 BFM 状态配置
在 [deploy/robots/g1/config/config.yaml](deploy/robots/g1/config/config.yaml) 的 `FSM.BFM_goal / FSM.BFM_reward / FSM.BFM_tracking` 中配置:
- `policy_dir`: BFM 模型目录
- `deploy_yaml`: 部署参数路径（相对 `policy_dir`）
- `onnx_model`: ONNX 文件路径（相对 `policy_dir`）
- `onnx_cuda`: 是否开启 CUDA（默认建议 `true`）
- `onnx_cuda_device`: CUDA 设备号
- `task_type`: `goal` / `reward` / `tracking`
- `latent_file`: 对应任务潜变量 `.npz`
- `gamepad_map`: 按需覆盖 `start_motion` / `next_latent` / `reset_state`
- 任务附加参数:
    - `goal.selected_goals`: 目标列表（可选）
    - `reward.selected_rewards_filter_z`: reward 与 z 索引过滤（可选）
    - `tracking.key/start/end/stop/gamma/window_size`

### 3. deploy.yaml 中 observations 配置规则（重点）
BFM 的 [deploy.yaml](../logs/g1/bfm/official/param/deploy.yaml) 使用双组观测，避免 YAML 同名 key 冲突:
- `observations.obs_base`: 当前帧观测
- `observations.obs_hist`: 历史观测

推荐顺序:
- `obs_base`: `joint_pos_rel -> joint_vel_rel -> projected_gravity -> base_ang_vel -> last_action`
- `obs_hist`: `last_action -> base_ang_vel -> joint_pos_rel -> joint_vel_rel -> projected_gravity`

说明:
- `history_length=1` 表示当前帧项
- 历史项使用 `history_length=4`（按模型导出配置）
- `State_BFM` 内会按 BFM 需要顺序拼接 `obs_base + obs_hist`，并自动处理历史帧方向与 Python 一致

### 4. 典型排查
- 报错 `Observation term 'xxx' is not registered`:
    - 检查 term 名是否为 C++ 已注册观测（如 `joint_pos_rel/joint_vel_rel/base_ang_vel/projected_gravity/last_action`）
- 切换状态后立即回 `Passive`:
    - 检查 `bad_orientation_check` 配置，调试时可先关闭
- 推理性能:
    - 运行日志会周期输出 `avg_infer_ms`，用于观察 ONNX 推理平均耗时
