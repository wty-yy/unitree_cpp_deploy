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

## OmniXtreme配置说明
### 1. 目录与文件约定
默认权重较大，下载[Google Drive - official_omnixtreme.tar.zst](https://drive.google.com/file/d/1ffYiU07X2I-bpAYFBqg3ekJ4VNndMIrL/view?usp=sharing)到`logs/g1/omnixtreme`文件夹下，解压后目录结构示例如下：
- 策略目录示例: `logs/g1/omnixtreme/official/`
- 基础策略: `exported/base_policy_trt.onnx`
- 残差策略: `exported/residual_policy.onnx`
- FK 模型: `exported/fk_trt.onnx`
- 部署参数: `params/deploy.yaml`
- 轨迹文件:
  - 单条轨迹: `exported/motions/motion.npz`
  - 多条轨迹: `exported/motions/*.npz`

### 2. config.yaml 中 OmniXtreme 状态配置
在 [deploy/robots/g1/config/config.yaml](deploy/robots/g1/config/config.yaml) 的 `FSM.OmniXtreme` 中配置:
- `policy_dir`: OmniXtreme 模型目录
- `deploy_yaml`: 部署参数路径，相对 `policy_dir`
- `base_model`: base policy ONNX 路径，相对 `policy_dir`
- `residual_model`: residual policy ONNX 路径，相对 `policy_dir`
- `fk_model`: 腰部 FK ONNX 路径，相对 `policy_dir`
- `motion_files`: 轨迹 `.npz` 列表，相对 `policy_dir`
- `onnx_cuda` / `onnx_tensorrt` / `onnx_cuda_device`: ONNX Runtime 执行后端
- `residual_scale`: 残差增益，默认 `1.0`
- `loop_trajectory`: 是否循环播放轨迹
- `root_body_index`: `body_quat_w` 中 root body 索引，用于初始 yaw 对齐
- `anchor_body_index`: `body_quat_w` 中 anchor body 索引，用于构造 `anchor_ori_6d`
- `gamepad_map.next_trajectory`: 切换轨迹按键
- `gamepad_map.previous_trajectory`: 反向切换轨迹按键
- `gamepad_map.reset_trajectory`: 重置当前轨迹按键
- `gamepad_map.toggle_execute`: 开始/暂停动作按键

控制相关常量统一放在 [logs/g1/omnixtreme/official/params/deploy.yaml](../logs/g1/omnixtreme/official/params/deploy.yaml) 的 `omnixtreme` 节中，当前为必填项:
- `pd_bias_joint_pos`
- `action_scale`
- `p_gains / d_gains`
- `envelope_x1 / envelope_x2 / envelope_y1 / envelope_y2`
- `friction_va / friction_fs / friction_fd`

当前默认按键:
- `FixStand / Velocity_Y / BFM_goal -> OmniXtreme`: `RT + X.on_pressed`
- `开始 / 暂停动作`: `B.on_pressed`
- `切换下一条轨迹`: `Y.on_pressed`
- `切换上一条轨迹`: `A.on_pressed`
- `重置当前轨迹`: `X.on_pressed`
- 进入 OmniXtreme 后默认处于暂停站立状态，暂停时仍使用当前轨迹首帧参考输入策略，按 `B` 后开始执行当前轨迹

### 3. deploy.yaml 与轨迹顺序要求
- [params/deploy.yaml](../logs/g1/omnixtreme/official/params/deploy.yaml) 中 `joint_ids_map` 需要与训练使用的 G1 URDF 顺序一致
- 当前默认顺序为:
  - `left_hip_pitch,left_hip_roll,left_hip_yaw,left_knee,left_ankle_pitch,left_ankle_roll`
  - `right_hip_pitch,right_hip_roll,right_hip_yaw,right_knee,right_ankle_pitch,right_ankle_roll`
  - `waist_yaw,waist_roll,waist_pitch`
  - `left_shoulder_pitch,left_shoulder_roll,left_shoulder_yaw,left_elbow,left_wrist_roll,left_wrist_pitch,left_wrist_yaw`
  - `right_shoulder_pitch,right_shoulder_roll,right_shoulder_yaw,right_elbow,right_wrist_roll,right_wrist_pitch,right_wrist_yaw`
- 轨迹 `.npz` 中 `joint_pos/joint_vel` 按 BeyondMimic motionlib 顺序导出，状态内会使用 `PERM/INV_PERM` 转到部署顺序
- `body_quat_w` 与 FK `rot` 输出按 `wxyz` 解释

### 4. 实机使用注意事项
- 进入 OmniXtreme 前，机器人姿态需尽量接近目标轨迹首帧，建议先进入 `FixStand`
- `motion_files` 建议先只放一条稳定轨迹，确认效果后再加入多条切换
- `action_clip` 当前不再参与 raw action 裁剪，动作保护主要依赖:
  - 关节限位
  - torque-speed envelope 回推的位置限制
  - friction compensation 前馈
- 若出现“姿态接近但动作发虚”，优先检查:
  - `params/deploy.yaml` 的 `joint_ids_map`
  - 轨迹首帧是否与真机站姿接近
  - `root_body_index / anchor_body_index` 是否与该批 motion 数据一致

### 5. 典型排查
- 报错 `Unknown FSM type OmniXtreme`:
  - 新增 `State_OmniXtreme.cpp` 后需要重新执行 `cmake -S . -B build`
- 切换状态时 ONNX 报输入输出错误:
  - 检查 `base_model / residual_model / fk_model` 是否与 `py_omnixtreme` 同源
- 动作方向明显错误:
  - 优先检查轨迹 `npz` 是否为该模型对应导出版本
- 可以推理但动作很弱:
  - 检查 `residual_scale`
  - 检查轨迹首帧是否过于保守或不匹配当前姿态
