<div align="center">
  <h1 align="center">Go2 RL CPP Deploy</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

本项目用于在 Unitree Go2 Edu 机器人 (搭载 Orin NX) 上部署强化学习 (RL) 策略，代码基于[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)修改，可通过网线或者机载电脑部署。

## 目录结构

```
unitree_rl_lab/
├── deploy/                 # 部署相关的代码
│   ├── include/            # 通用头文件 (FSM, Isaac Lab 接口等)
│   ├── robots/go2/         # Go2 机器人的主程序、CMakeLists 和配置
│   └── thirdparty/         # 第三方库 (onnxruntime, json)
├── logs/                   # 存放训练好的 RL 策略模型
└── README.md
```

## 依赖项

在编译和运行之前，请确保开发环境 (Orin NX) 已安装以下依赖项：

- **[unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)**: Unitree 机器人的开发 SDK。
- **Boost**: 用于程序选项解析 (`program_options`)。
- **yaml-cpp**: 用于读取配置文件。
- **Eigen3**: 矩阵运算库。
- **fmt**: 格式化输出库。
- **onnxruntime**:
   - Orin NX: 需下载[onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2)解压到`deploy/thirdparty/`文件夹中
   - x64 Linux: 需下载[onnxruntime-linux-x64-1.23.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz)解压到`deploy/thirdparty/`文件夹中，修改[{ROBOT}/CMakeLists.txt](deploy/robots/go2/CMakeLists.txt)中的onnx链接路径(解开相应注释)

> 如果使用Orin NX机载电脑部署，则安装aarch64版本的onnxruntime；如果使用x64 Linux电脑链接网线部署，则安装x64版本的onnxruntime并修改ONNX链接路径。

## 编译步骤

1. 进入 Go2 机器人部署目录：
   ```bash
   cd deploy/robots/go2
   ```

2. 创建编译目录：
   ```bash
   mkdir build && cd build
   ```

3. 运行 CMake 并编译：
   ```bash
   cmake .. && make -j8
   ```

## 运行指南

编译完成后，在 `build` 目录下运行生成的 `go2_ctrl` 可执行文件。

### 基本用法

#### 命令行参数

- `-h, --help`: 显示帮助信息。
- `-v, --version`: 显示版本信息。
- `--log`: 开启日志记录 (日志将保存在 `deploy/robots/go2/log/` 目录下)。
- `-n, --network <interface>`: 指定用于 DDS 通信的网络接口名称 (例如 `eth0`, `wlan0`)。如果不指定，将使用默认接口。

#### 真机启动
将Go2启动，进入站立状态后 **[L2+A]** 两次让机械狗趴下，使用手机App连接上后，依次点击：设置-服务状态，关闭`mcf/*`服务，关闭官方控制程序，避免控制冲突。

```bash
sudo ./go2_ctrl [选项]
```

**示例：**

在NX上启动, 下位机网卡假设为 `eth0`
```bash
./go2_ctrl -n eth0
```

#### 仿真启动

使用xbox数据协议的手柄即可控制机器人, 如需其他协议, 参考unitree_mujoco修改手柄配置文件

下载并编译[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)中的`simulate/`内容，并配置`simulate/config.yaml`中`domain_id: 0`, `use_joystick: 1`
```bash
./simulate/build/unitree_mujoco  # 启动仿真
./go2_ctrl -n lo  # 启动控制
```

### 操作流程

1. 启动程序后，控制台将显示 "Waiting for connection to robot..."
2. 确保机器人上电并连接正常，程序连接成功后会显示 "Connected to robot."
3. **进入站立模式**：按下手柄上的 **[L2 + A]** 组合键，机器人将进入 `FixStand` 模式并站立
4. **开始 RL 控制**：按下手柄上的 **[Start + Up/Down/Left/Right]** 键，机器人将切换到相应的 `policy_dir_[up,down,left,right]` 控制模型，开始执行 RL 策略
5. **模型切换**：在运行过程中，可以随时通过按下 **[Start + 方向键]** 切换到不同的 RL 模型
6. **固定指令执行**：按下手柄上的 **[L2 + Y]** 组合键，机器人将开始执行预设的固定指令（如配置文件中所设），再次按下该组合键将停止固定指令执行
7. **进入阻尼模式**：按下手柄上的 **[L2 + B]** 组合键，机器人将进入阻尼模式，停止 RL 控制

https://github.com/user-attachments/assets/56375e44-5ac1-42b0-a268-7837c2857287

## 配置说明

配置文件位于 [deploy/robots/go2/config/config.yaml](./deploy/robots/go2/config/config.yaml), 包含功能：
1. 多模型选择功能, 配置`Velocity/policy_dir_up/down/left/right`, 分别指定四个模型的路径, 按`Start + 方向键`切换模型
2. 实时记录运行时数据, 配置`Velocity/logging: true`, 设置记录频率 `logging_dt`, 默认100Hz, 存储位置在模型文件夹下, 例如 `./logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745/logs/`
3. 固定指令执行功能, 配置`Velocity/fixed_command/enabled: true`, 设置固定指令值`command`, 以及持续时间`duration`(可选), 按`L2 + Y`启动/停止固定指令执行

### 更换策略模型

要更换使用的 RL 策略，请修改 `config.yaml` 中的 `Velocity.policy_dir` 字段

```yaml
Velocity:
  # 策略模型路径 (相对于项目根目录或绝对路径)
  policy_dir_up: ../../../logs/rsl_rl/go2_moe_cts_expert_goal_137000_0.6745  # best model
  policy_dir_down: ../../../logs/rsl_rl/go2_moe_cts_fast_flat_v4_fz_54k  # fast model
  policy_dir_left: ../../../logs/rsl_rl/kaiwu2025_v6-2_124004  # not bad model
  policy_dir_right: null
```

指定的目录结构应包含：
- `exported/policy.onnx`: 导出的 ONNX 策略模型。
- `params/deploy.yaml`: 对应的部署参数。

### 修改控制参数

你也可以在 `config.yaml` 中调整 `FixStand` 模式下的 PD 参数 (`kp`, `kd`) 以及目标关节角度 (`qs`)。

### 修改意外终止参数

根据base_link的z轴与重力夹角大小判断是否处以意外状态，中止控制程序，默认2rad。

在[State_RLBase.cpp](deploy/robots/go2/src/State_RLBase.cpp)文件中找到`bad_orientation`中第二个参数修改rad阈值。
