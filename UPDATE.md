# UPDATE
## 20260326 v0.6.1
**新增Go2 go2_moe_cts_self_103.5k_0.6669模型**
1. 通过Start+Left切换到该模型，通过开启自碰撞后训练，移动时不易绊脚
## 20260316
### v0.6
**新增OmniXtreme模型**
1. 新增 [State_OmniXtreme.h](deploy/robots/g1/include/State_OmniXtreme.h) / [State_OmniXtreme.cpp](deploy/robots/g1/src/State_OmniXtreme.cpp):
    - 接入 G1 OmniXtreme base policy + residual policy + FK ONNX 推理
    - 支持从 `motion npz` 读取参考轨迹，并在状态内完成 `PERM/INV_PERM` 顺序转换
    - 支持初始 yaw 对齐、anchor 6D 姿态构造、残差观测拼接、真机 PD/friction/envelope 控制输出
2. FSM 与配置集成:
    - 在 [config.yaml](deploy/robots/g1/config/config.yaml) 新增 `OmniXtreme` 状态
    - 新增 `RT + X.on_pressed` 进入 OmniXtreme
    - `Y.on_pressed` 切换轨迹, `X.on_pressed` 重置当前轨迹
3. 文档更新:
    - 在 [docs/g1_setup_zh.md](docs/g1_setup_zh.md) 增加 OmniXtreme 配置说明
4. 使用注意:
    - `params/deploy.yaml` 的 `joint_ids_map` 必须与训练时 G1 URDF 顺序一致
    - 轨迹 `body_quat_w` 与 FK `rot` 输出按 `wxyz` 解释
    - raw action 不做额外 `[-1, 1]` 裁剪，动作保护依赖关节限位与 envelope
    - 进入状态前机器人姿态应尽量接近目标轨迹首帧
### v0.5.1
**新增G1 Loco v0.0.5.1模型，降低踏地的声音**
1. 删除之前的v0.0.5运控模型，新增v0.0.5.1，并在配置中加入对BFM-Zero支持，在摔倒后可以通过BFM重新站立后进入运控
2. 切换运控RB+Y，切换BFM RT+Y

## 20260305 v0.5
**新增G1 BFM-Zero部署支持Onnxruntime CUDA加速**
1. 新增 [State_BFM.h](deploy/robots/g1/include/State_BFM.h) / [State_BFM.cpp](deploy/robots/g1/src/State_BFM.cpp):
    - 支持 G1 BFM-Zero 三种任务模式: `goal` / `reward` / `tracking`
    - 支持从 `npz` 读取潜变量 (`goal_inference` / `reward_inference` / `tracking_inference`)
    - ONNX Runtime 推理默认优先使用 CUDA，自动回退 CPU
    - 新增推理线程内统计并打印区间平均推理耗时 `avg_infer_ms`
2. FSM 集成:
    - 在 [main.cpp](deploy/robots/g1/main.cpp) 注册 `State_BFM`
    - 在 [config.yaml](deploy/robots/g1/config/config.yaml) 增加 `BFM_goal` / `BFM_reward` / `BFM_tracking` 配置项
    - 新增 `Velocity` 与 `BFM_*` 之间两两切换（与 `FixStand` 触发按键保持一致）
3. BFM 手柄控制逻辑与 Python 对齐:
    - 进入 BFM 状态后默认启用策略
    - 保留 `start` / `next` / `reset` 三类交互
4. BFM 观测构建重构:
    - 从“完全手写历史缓存”改为复用 FSM `ObservationManager`
    - BFM `deploy.yaml` 的 `observations` 改为 `obs_base` + `obs_hist` 双组，解决同名 key 冲突
    - `State_BFM` 中按固定顺序拼接 `obs_base + obs_hist`，并将历史帧顺序转换为与 Python 一致
5. 文档更新:
    - 完善 [docs/g1_setup_zh.md](docs/g1_setup_zh.md) 的 G1 BFM 使用方法与配置说明

## 20260301 v0.4
**重构 FSM 框架: 从硬编码转为 YAML 配置驱动，支持声明式状态注册和跳转**

1. [BaseState.h](deploy/include/FSM/BaseState.h):
    - 新增 `FsmFactory`/`FsmMap` 工厂类型定义和 `getFsmMap()` 全局工厂注册表
    - 新增 `REGISTER_FSM(Derived)` 宏，在头文件中自动注册 FSM 状态类到工厂
2. [CtrlFSM.h](deploy/include/FSM/CtrlFSM.h):
    - 新增 `CtrlFSM(YAML::Node cfg)` 构造函数，从配置文件 `FSM._` 节自动读取状态声明（id/type），通过工厂创建状态实例
    - 分离 `start()` 方法，支持先注册再启动
    - 状态管理从裸指针改为 `std::shared_ptr<BaseState>`
3. [FSMState.h](deploy/include/FSM/FSMState.h):
    - 移除硬编码的 `L2+B → Passive` 转换逻辑
    - 新增从 YAML `transitions` 节读取跳转条件，使用 DSL 解析器将字符串表达式编译为运行时谓词
4. 新增 [unitree_joystick_dsl.hpp](deploy/include/unitree_joystick_dsl.hpp):
    - 手柄按键组合 DSL 解析器，支持 `+`(AND)、`|`(OR)、`!`(NOT)、`()`分组、`.on_pressed`/`.on_released`/`.pressed` 状态检测、`LT(2s)` 长按检测
5. [State_FixStand.h](deploy/include/FSM/State_FixStand.h) / [State_Passive.h](deploy/include/FSM/State_Passive.h) / [State_RLBase.h](deploy/include/FSM/State_RLBase.h):
    - 构造函数统一为 `(int state, std::string state_string)` 签名
    - 添加 `REGISTER_FSM` 宏注册
    - `State_RLBase` 移除 `policy_key`/`config_name` 参数，改为从 `FSM.{state_string}.policy_dir` 读取
6. robots/g1、robots/go2 适配:
    - `Types.h` 移除 `FSMMode` 枚举，状态 ID 改由 config.yaml 定义
    - `main.cpp` 用 `CtrlFSM(param::config["FSM"])` + `fsm->start()` 替代手动创建和注册
    - `State_RLBase.cpp` 适配新构造函数，用 `FSMStringMap.right.at("Passive")` 替代枚举
    - `config.yaml` 新增 `_` 节声明状态和 `transitions` 节定义跳转条件
7. 详细配置使用方法见 [docs/robot_params.md](docs/robot_params.md)

## 20260216 v0.3
**加入g1人形机器人适配更新，后续上传稳定运控模型**
1. [manager_term_cfg.h](deploy/include/isaaclab/manager/manager_term_cfg.h):
    - 新增 `get(int n)` 方法返回单帧数据, 修复 gym-style history 叠帧 bug (原 `get()` 返回全帧拼接, 外层再循环导致 5×5 膨胀)
    - `ObsFunc` 签名增加 `YAML::Node params` 参数
    - scale/clip 处理从 `get()` 移到 `add()` 中, 新增 `scale_first` 选项
    - 新增 `params` 成员存储 term 配置, 新增 `size()` 方法
2. [observation_manager.h](deploy/include/isaaclab/manager/observation_manager.h):
    - 支持 observation group: 底层存储改为 `unordered_map<string, vector<ObservationTermCfg>>`, `compute()` 返回 `unordered_map<string, vector<float>>`
    - gym-history 分支改用 `term.get(h)` 取单帧, 修复维度膨胀
    - `REGISTER_OBSERVATION` 宏签名增加 `params` 参数
    - 新增 `_prepare_group_terms()`, 自动识别单 group (flat) / 多 group 配置格式
3. [algorithms.h](deploy/include/isaaclab/algorithms/algorithms.h):
    - `act()` / `forward()` 入参从 `vector<float>` 改为 `unordered_map<string, vector<float>>`
    - `OrtRunner` 动态检测所有模型 input 名称和 shape, 不再硬编码 `input_names={"obs"}`
    - 支持多 input tensor 推理
4. [observations.h](deploy/include/isaaclab/envs/mdp/observations/observations.h):
    - 所有 term 处理函数签名增加 `YAML::Node params` 参数, 对应 `item["params"]` 内容, 从中读取配置参数
    - `joint_pos`/`joint_pos_rel`/`joint_vel_rel` 改为从 `params["asset_cfg"]["joint_ids"]` 读取 `joint_ids`
    - `gait_phase` 改为从 `params["period"]` 读取周期
5. 新增[Observation Group](./docs/obs_group.md)使用说明

## 20260211 v0.2.2
1. SportModeState 高层估计，真机上没有用删掉
## 20260210 v0.2.1
1. 修改moe模型命名为go2_moe_cts_137k_0.6739
## 20260125 v0.2
1. 添加固定指令执行功能:
    - 在[config.yaml](deploy/robots/go2/config/config.yaml)的`fixed_command.enabled`字段打开该功能, 并设置固定指令值
    - 在真机上通过按键组合`L2 + Y`启动固定指令执行，并再次按下该组合键停止执行固定指令
    - `fixed_command`如果设置了默认执行时间`duration`，则在启动后会持续执行该指令一段时间后自动停止
2. 新增四种模型选择, 删除原有的`Start`启动模型按键:
    - Velocity_Up: 通过按键组合`Start + Up`切换到该模型, 使用`policy_dir_up`中的模型文件
    - Velocity_Down: 通过按键组合`Start + Down`切换到该模型, 使用`policy_dir_down`中的模型文件
    - Velocity_Left: 通过按键组合`Start + Left`切换到该模型, 使用`policy_dir_left`中的模型文件
    - Velocity_Right: 通过按键组合`Start + Right`切换到该模型, 使用`policy_dir_right`中的模型文件
3. 进入模型方法: L2 + A站立, start + 方向键选择模型, 随时可start + 方向键切换模型, L2 + B进入阻尼模式
## 20251208 v0.1
1. 新增运行时数据记录功能:
    - 在[config.yaml](deploy/robots/go2/config/config.yaml)的`logging`打开该功能, 及记录频率`logging_dt`
    - 实现 `DataLogger` 类, 支持CSV格式数据保存存储在 `{policy_dir}/logs` 下, `State_RLBase` 中集成日志记录, 涵盖:
    ```bash
    "time": 当前程序启动时长
    "unix_time": 时间戳 (精度0.1s)
    "wall_time": H:M:S 时间字符串 (精度0.1s)
    "q_des": 目标关节位置 12
    "q": 实际关节位置 12 
    "dq": 实际关节速度 12
    "tau": 关节力矩 (PD计算出的力矩) 12
    "temperatures": 全部关节电机温度 12
    "imu_rpy": imu的roll,pitch,yaw 3
    "imu_acc": imu加速度 (评估冲击) 3
    "ang_vel": imu陀螺仪 (角速度) 3
    "foot_force": 足端接触力 4
    "foot_contacts": 足端接触状态 4

    # MoE Model
    "weight": 8
    "latent": student encoder输出的潜空间向量 32
    "cmd_vel_x_no_scale": 范围(0,1)的x线速度指令 1
    "cmd_vel_y_no_scale": 范围(0,1)的y线速度指令 1
    "cmd_ang_z_no_scale": 范围(0,1)的z角速度指令 1

    # SportModeState 高层估计 （没有用删掉）
    "position": 里程计估计的位置 3
    "velocity": 里程计估计的速度 3
    ```
    - 修改扩大指令范围逻辑, 只需修改`base_velocity.ranges`, 即将手柄的(-1,1)缩放到该指令范围, 无需再对`velocity_commands.scale`修改
2. ONNX推理逻辑修改: 修改 `OrtRunner` 类，新增 `forward()` 接口，支持多输出推理，解决隐藏状态提取问题
