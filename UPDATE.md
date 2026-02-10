# UPDATE
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

    # SportModeState 高层估计
    "position": 里程计估计的位置 3
    "velocity": 里程计估计的速度 3
    ```
    - 修改扩大指令范围逻辑, 只需修改`base_velocity.ranges`, 即将手柄的(-1,1)缩放到该指令范围, 无需再对`velocity_commands.scale`修改
2. ONNX推理逻辑修改: 修改 `OrtRunner` 类，新增 `forward()` 接口，支持多输出推理，解决隐藏状态提取问题
