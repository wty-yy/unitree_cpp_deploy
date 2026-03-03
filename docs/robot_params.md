# FSM 配置说明

## 概述

FSM（有限状态机）通过 `config.yaml` 配置文件驱动，支持：
- 声明式注册 FSM 状态及其 ID
- 通过 DSL（Domain Specific Language） 表达式定义状态间的跳转条件（手柄按键组合）
- 通过 `type` 字段复用同一状态类创建多个实例

## 配置文件结构

```yaml
FSM:
  _:           # 状态声明区（必须）
    状态名:
      id: <int>           # 状态唯一 ID（必须，不可重复）
      type: <string>      # 状态类型（可选，默认为状态名）

  状态名:      # 状态参数区
    transitions:          # 跳转条件（可选）
      目标状态名: <DSL表达式>
    # ... 其他状态特定参数
```

### `_` 状态声明区

`FSM._` 节用于声明所有启用的 FSM 状态。程序启动时按此节的顺序创建状态实例，**第一个状态为初始状态**（通常是 `Passive`）。

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `id` | int | 是 | 状态的唯一标识符，用于内部查找，不可重复 |
| `type` | string | 否 | 对应的状态类名（不含 `State_` 前缀）。省略时默认为状态名本身 |

#### type 字段说明

`type` 决定了使用哪个 C++ 类来实例化该状态。例如：

```yaml
_:
  Velocity:          # 状态名为 "Velocity"
    id: 3
    type: RLBase     # 使用 State_RLBase 类实例化
```

当多个状态需要复用同一个类时，必须指定 `type`：

```yaml
_:
  Velocity:
    id: 3
    type: RLBase           # State_RLBase 实例，名为 "Velocity"
  Mimic_Dance:
    id: 101
    type: Mimic            # State_Mimic 实例，名为 "Mimic_Dance"
  Mimic_Gangnam_Style:
    id: 102
    type: Mimic            # State_Mimic 实例，名为 "Mimic_Gangnam_Style"
```

注释掉某个状态即可禁用：

```yaml
_:
  # Velocity_Right:   # 注释掉即禁用
  #   id: 7
  #   type: RLBase
```

### transitions 跳转条件

每个状态的 `transitions` 节定义了从该状态跳转到其他状态的条件。格式为 `目标状态名: DSL表达式`。

```yaml
FixStand:
  transitions:
    Passive: LT + B.on_pressed        # 按 LT + B 跳转到 Passive
    Velocity: RB + X.on_pressed       # 按 RB + X 跳转到 Velocity
```

**注意事项：**
- 跳转条件按声明顺序检查，匹配到第一个满足的条件即触发跳转
- 所有状态自动注册超时检测（`isTimeout()`），超时后自动跳转到 `Passive`
- 目标状态必须在 `_` 节中声明过，否则会打印警告并跳过

---

## 手柄 DSL 语法

DSL（Domain Specific Language）用于描述手柄按键组合条件，支持以下语法：

### 基本按键

| 表达式 | 含义 |
|--------|------|
| `A` | A 键被按下（持续） |
| `A.on_pressed` | A 键刚按下（单帧触发） |
| `A.on_released` | A 键刚松开（单帧触发） |
| `A.pressed` | 等同于 `A`，显式写法 |

### 支持的按键名称

| 按键 | 名称（不区分大小写） |
|------|---------------------|
| 功能键 | `A`, `B`, `X`, `Y` |
| 肩键 | `LB`（左肩键）, `RB`（右肩键）, `LT`（左扳机）, `RT`（右扳机） |
| 摇杆 | `LX`, `LY`, `RX`, `RY` |
| 摇杆按下 | `LS`, `RS` |
| 方向键 | `up`, `down`, `left`, `right` |
| 系统键 | `start`, `back` |
| 扩展键 | `F1`, `F2` |

### 组合操作

| 操作符 | 含义 | 示例 |
|--------|------|------|
| `+` | AND（同时满足） | `LT + B.on_pressed` |
| `\|` | OR（满足其一） | `A.on_pressed \| B.on_pressed` |
| `!` | NOT（取反） | `!A + B` |
| `()` | 分组 | `(A + B) \| (X + Y)` |

### 长按检测

在按键名后加 `(Ns)` 表示长按 N 秒：

| 表达式 | 含义 |
|--------|------|
| `LT(2s)` | LT 按住超过 2 秒 |
| `LT(2s) + down.on_pressed` | LT 长按 2 秒 + 方向键下刚按下 |
| `(LT(2s) + RT(2s)) + A` | LT 和 RT 同时长按 2 秒 + A 按下 |

时间单位支持 `s`、`sec`、`secs`。

### 完整示例

```yaml
transitions:
  # 基本跳转
  FixStand: LT + up.on_pressed
  Passive: LT + B.on_pressed

  # 长按触发
  Mimic_Dance: LT(2s) + down.on_pressed

  # 多条件
  Velocity: RB + X.on_pressed

  # 复杂组合
  Special: ((LT(1s) + up) | (RB + X.on_pressed)) + !Y
```

---

## 内置状态类型

### State_Passive

阻尼模式，关节只有阻尼无刚度。

```yaml
Passive:
  transitions:
    FixStand: LT + up.on_pressed
  mode: [1,1,1, ...]    # 电机模式（可选）
  kd: [3,3,3, ...]      # 阻尼系数
```

### State_FixStand

固定站立，通过线性插值从当前位置运动到目标位置。

```yaml
FixStand:
  transitions:
    Passive: LT + B.on_pressed
    Velocity: RB + X.on_pressed
  kp: [100, 100, ...]   # 关节刚度
  kd: [2, 2, ...]       # 关节阻尼
  ts: [0, 3]            # 时间节点（秒）
  qs:                    # 对应时间节点的目标关节角度
    - []                 # 第一个为空（运行时自动填充当前位置）
    - [-0.1, 0, ...]    # 目标位置
```

### State_RLBase

强化学习策略执行。

```yaml
Velocity:
  transitions:
    Passive: LT + B.on_pressed
  policy_dir: ../../../logs/g1/g1_moe_cts_v0.0.5_seed1_50k  # 策略文件目录
  # 可选参数
  logging: false         # 是否记录运行数据
  logging_dt: 0.01       # 记录间隔（秒）
  fixed_command:         # 固定指令配置（可选）
    enabled: true
    lin_vel_x: 1.0
    lin_vel_y: 0.0
    ang_vel_z: 0.0
    duration: 3.0        # 持续时间（秒），null 为无限
```

`policy_dir` 目录结构要求：
```
policy_dir/
├── exported/
│   └── policy.onnx      # ONNX 模型文件
└── params/
    └── deploy.yaml       # 部署配置文件
```

---

## 完整配置示例

### G1 29-DOF

```yaml
FSM:
  _:
    Passive:
      id: 1
    FixStand:
      id: 2
    Velocity:
      id: 3
      type: RLBase

  Passive:
    transitions: 
      FixStand: LT + up.on_pressed
    mode: [1,1,1,1,1,1, 1,1,1,1,1,1, 1,1,1, 1,1,1,1,1,1,1, 1,1,1,1,1,1,1]
    kd: [3,3,3,3,3,3, 3,3,3,3,3,3, 3,3,3, 3,3,3,3,3,3,3, 3,3,3,3,3,3,3]

  FixStand:
    transitions: 
      Passive: LT + B.on_pressed
      Velocity: RB + X.on_pressed
    kp: [100,100,100,150,40,40, 100,100,100,150,40,40, 200,200,200, 40,40,40,40,40,40,40, 40,40,40,40,40,40,40]
    kd: [2,2,2,4,2,2, 2,2,2,4,2,2, 5,5,5, 10,10,10,10,10,10,10, 10,10,10,10,10,10,10]
    ts: [0, 3]
    qs: [[], [-0.1,0,0,0.3,-0.2,0, -0.1,0,0,0.3,-0.2,0, 0,0,0, 0,0.25,0,0.97,0.15,0,0, 0,-0.25,0,0.97,-0.15,0,0]]

  Velocity:
    transitions: 
      Passive: LT + B.on_pressed
    policy_dir: ../../../logs/g1/g1_moe_cts_v0.0.5_seed1_50k
```

操作流程：`Passive` → [LT + Up] → `FixStand` → [RB + X] → `Velocity`，任意状态 [LT + B] 回到 `Passive`

### Go2 多策略切换

```yaml
FSM:
  _:
    Passive:
      id: 1
    FixStand:
      id: 2
    Velocity_Up:
      id: 4
      type: RLBase
    Velocity_Down:
      id: 5
      type: RLBase
    Velocity_Left:
      id: 6
      type: RLBase

  Passive:
    transitions: 
      FixStand: LT + A.on_pressed

  FixStand:
    transitions: 
      Passive: LT + B.on_pressed
      Velocity_Up: start + up.on_pressed
      Velocity_Down: start + down.on_pressed
      Velocity_Left: start + left.on_pressed

  Velocity_Up:
    transitions: 
      Passive: LT + B.on_pressed
      Velocity_Down: start + down.on_pressed
      Velocity_Left: start + left.on_pressed
    policy_dir: ../../../logs/go2/go2_moe_cts_137k_0.6739

  Velocity_Down:
    transitions: 
      Passive: LT + B.on_pressed
      Velocity_Up: start + up.on_pressed
      Velocity_Left: start + left.on_pressed
    policy_dir: ../../../logs/go2/go2_moe_cts_fast_flat_v4_fz_54k

  Velocity_Left:
    transitions: 
      Passive: LT + B.on_pressed
      Velocity_Up: start + up.on_pressed
      Velocity_Down: start + down.on_pressed
    policy_dir: ../../../logs/go2/kaiwu2025_v6-2_124004
```

操作流程：`Passive` → [LT + A] → `FixStand` → [Start + 方向键] 选择策略，策略之间可通过 [Start + 方向键] 直接互相切换

---

## 自定义状态类

实现新的 FSM 状态需要：

1. 继承 `FSMState` 并实现 `enter()`/`run()`/`exit()` 方法
2. 在头文件末尾调用 `REGISTER_FSM(State_YourType)` 注册到工厂
3. 在 `main.cpp` 中 `#include` 该头文件（确保注册宏被执行）

```cpp
// State_MyState.h
#pragma once
#include "FSMState.h"

class State_MyState : public FSMState
{
public:
    State_MyState(int state, std::string state_string = "MyState") 
    : FSMState(state, state_string) 
    {
        // 从配置读取参数
        auto cfg = param::config["FSM"][state_string];
        // ...
    }

    void enter() override { /* 进入状态时执行 */ }
    void run() override   { /* 每个控制周期执行 */ }
    void exit() override  { /* 退出状态时执行 */ }
};

REGISTER_FSM(State_MyState)
```

在 config.yaml 中使用：

```yaml
FSM:
  _:
    MyCustomState:
      id: 10
      type: MyState     # 对应 State_MyState 类
  MyCustomState:
    transitions:
      Passive: LT + B.on_pressed
    # 自定义参数 ...
```
