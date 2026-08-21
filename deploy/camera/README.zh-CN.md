# Camera Tools

[English](README.md)

独立的 D435i 深度采集、DDS 发布、预览和视频录制工具。

## 组件

| 程序 | 功能 |
| --- | --- |
| `d435i_depth_publisher` | RealSense 深度采集与 `32FC1` DDS 发布 |
| `depth_camera_viewer` | DDS 深度预览与 H.265/MKV 视频录制 |

默认 DDS topic：

```text
rt/front_depth_observation
cpp_viewer/record_switch
rt/wirelesscontroller
```

## 编译

从 `deploy/camera/CMakeLists.txt` 同时编译两个程序：

```bash
# 在仓库根目录执行
cmake -S deploy/camera -B deploy/camera/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/camera/build -j
```

输出文件：

```text
deploy/camera/build/d435i/d435i_depth_publisher
deploy/camera/build/viewer/depth_camera_viewer
```

## 使用

一键编译并启动：

```bash
./deploy/camera/run.sh --build
```

使用已有构建启动：

```bash
./deploy/camera/run.sh
```

脚本同时显示两路输出。`Ctrl+C` 或任一程序退出时，另一个程序同步关闭。

单独启动：

```bash
./deploy/camera/build/d435i/d435i_depth_publisher
./deploy/camera/build/viewer/depth_camera_viewer
```

Viewer 控制：

| 输入 | 操作 |
| --- | --- |
| 终端 `R` | 开始或停止录制 |
| 终端 `Q` | 退出 |
| 窗口 `R` | 开始或停止录制 |
| 窗口 `Esc` | 退出 |
| DDS `ON` | 开始录制 |
| DDS `OFF` | 停止录制 |
| Go2 / Unitree MuJoCo `Select+A` | 开始或停止录制 |

录制文件保存到：

```text
deploy/camera/videos/depth_YYYY-MM-DD_HH-MM-SS.mkv
```

## 配置

D435i 配置：

```text
deploy/camera/d435i/config/config.yaml
```

Viewer 配置：

```text
deploy/camera/viewer/config/config.yaml
```

两个配置中的 DDS `domain_id` 和 `interface` 必须一致。PC 本地测试默认使用 `lo`，机器人部署时按实际网卡修改。

默认深度链路：

```text
D435i 640x360 Z16 @ 30 Hz
-> 米制 float32
-> rt/front_depth_observation 32FC1 @ 15 Hz
-> Viewer 灰度预览 / H.265 录制
```

## 实现

- Publisher 负责相机采集和 DDS 发布。
- Viewer 使用最新 DDS 深度帧，按 `0-2 m` 映射为 8-bit 灰度图。
- FFmpeg 使用 `libx265`、`preset=fast`、`CRF=26` 和 MKV 容器。
- `cpp_viewer/record_switch` 使用 `std_msgs::msg::dds_::String_`，仅接受 `ON` 和 `OFF`。
- Viewer 订阅 `rt/wirelesscontroller` 并检测 `Select+A` 组合键上升沿。
- 视频用于可视化调试，不保留原始 `32FC1` 米制深度精度。
