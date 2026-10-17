<div align="center">
  <h1>Go2 RL CPP Deploy</h1>
  <p><a href="README.md">English</a> | <strong>中文</strong></p>
</div>

在 Unitree Go2 Edu 上部署强化学习策略，可在 Orin NX 上运行，也可通过以太网连接外部电脑。代码基于 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 修改。

## 安装与编译

```bash
# 安装依赖和 Git LFS
sudo apt install git-lfs libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libfmt-dev libspdlog-dev
git lfs install

# 拉取 go2 分支及五个 ONNX 策略
git clone --branch go2 --single-branch https://github.com/wty-yy/unitree_cpp_deploy.git
cd unitree_cpp_deploy
git lfs pull
```

安装 [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2)，将对应架构的 ONNX Runtime 解压到 `deploy/thirdparty/`。

| 部署平台 | ONNX Runtime |
|---|---|
| Orin NX（默认） | [aarch64 1.16.0](https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.16.0/onnxruntime-linux-aarch64-gpu-1.16.0.tar.bz2) |
| x64 Linux | [x64 1.24.2](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-gpu-1.24.2.tgz) |

x64 部署需同时切换 [CMakeLists.txt](deploy/robots/go2/CMakeLists.txt) 中的 include 和 lib 路径。

```bash
cmake -S deploy/robots/go2 -B deploy/robots/go2/build
cmake --build deploy/robots/go2/build -j4
```

## 运行

真机通过手机 App 关闭 `mcf/*` 服务，确认官方控制程序已停止后启动。

```bash
# Orin NX，网卡按实际连接修改
./deploy/robots/go2/build/go2_ctrl -n eth0
# unitree_mujoco 仿真：domain_id: 0，use_joystick: 1
./deploy/robots/go2/build/go2_ctrl -n lo
# 命令行帮助
./deploy/robots/go2/build/go2_ctrl --help
```

| 按键 | 功能 |
|---|---|
| L2 + A | Passive → 站立；策略 → 蹲下；蹲下 → 站立 |
| Start + Up / Down / Left / Right | 从站立或其他策略切换到对应策略 |
| L2 + B | 切换到 Passive 阻尼模式 |
| 按住 RT | 使用当前模型定义的 boost 指令范围 |
| L2 + Y | 启停固定指令，需先启用 `fixed_command.enabled` |

蹲下轨迹完成后自动进入 Passive。固定指令默认关闭。按键与状态参数见 [config.yaml](deploy/robots/go2/config/config.yaml)。

## 策略

五个策略位于 `logs/go2/`，每个包含 `exported/policy.onnx` 和 `params/deploy.yaml`。ONNX 通过 Git LFS 管理。

| 状态 | 策略目录（相对于 logs/go2） |
|---|---|
| Velocity_Up | `go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_74k_0.687` |
| Velocity_Down | `go2_rl_gym/go2_moe_cts_164k_0.6715` |
| Velocity_Left | `go2_rl_lab/go2_mixed_moe_cts_v1.3_8192_89k` |
| Velocity_Right | `go2_rl_lab/go2_mixed_moe_cts_v1.3.2resume_82.5k_0.6929` |
| 未绑定按键 | `go2_rl_robotlab/go2_moe_cts_v4.2_176k_0.6984` |

`go2_rl_robotlab` 作为额外 checkpoint 保留。使用时将对应状态的 `policy_dir` 指向该目录。

普通速度范围由各模型 `params/deploy.yaml` 中的 `commands.base_velocity.ranges` 定义。Up、Left 和 Right 还配置了 `boost.ranges`：

| 状态 | 纵向（m/s） | 横向（m/s） | 转向（rad/s） |
|---|---|---|---|
| Velocity_Up | `[-4.5, 4.2]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` |
| Velocity_Left | `[-4.0, 4.0]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` |
| Velocity_Right | `[-4.5, 3.0]` | `[-0.5, 0.5]` | `[-1.0, 1.0]` |

按住 RT 使用 boost 范围，松开恢复普通范围；可通过 `boost_button` 修改按键，不配置时默认 RT。Down 和额外保留的 `go2_rl_robotlab` 策略未配置 boost。固定指令执行期间忽略 boost。

策略目录相对于 `deploy/robots/go2/` 解析。替换模型时同步更新 `policy.onnx` 和 `deploy.yaml`。

## 配置与工具

[config.yaml](deploy/robots/go2/config/config.yaml) 末尾的可选参数位于 `Velocity_Right` 下，只作用于该状态。要在其他策略中使用，将对应配置复制到 `FSM.Velocity_Up/Down/Left` 下。

| 配置 | 功能 |
|---|---|
| `velocity_command_damping` | 纵向速度指令阻尼及加减速限制 |
| `logging` | `enabled` 控制日志开关，`dt` 设置采样间隔（秒）；CSV 保存到策略目录的 `logs/` |
| `fixed_command` | 通过 L2 + Y 启停预设速度指令 |
| `joint_pos_udp` | 通过 UDP 发送 12 个关节的实测位置 |

FSM 和按键配置见 [robot_params.md](docs/robot_params.md)，观测分组见 [obs_group.md](docs/obs_group.md)。[udp_test.py](scripts/udp_test.py) 提供 UDP 接收示例，`deploy/tools/` 包含日志和关节位置可视化脚本，`resources/go2.urdf` 为 Go2 模型。

## 许可证

项目自有代码采用 [MIT 许可证](LICENSE)。Unitree 派生代码的 Apache 2.0 条款见 [Unitree RL Lab 许可证](docs/licenses/unitree-rl-lab-license.txt)。
