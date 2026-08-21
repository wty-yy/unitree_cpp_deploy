# Camera Tools

[简体中文](README.zh-CN.md)

Standalone tools for D435i depth capture, DDS publishing, preview, and video recording.

## Components

| Program | Purpose |
| --- | --- |
| `d435i_depth_publisher` | RealSense depth capture and `32FC1` DDS publishing |
| `depth_camera_viewer` | DDS depth preview and H.265/MKV video recording |

Default DDS topics:

```text
rt/front_depth_observation
cpp_viewer/record_switch
rt/wirelesscontroller
```

## Build

Build both programs from `deploy/camera/CMakeLists.txt`:

```bash
# Run from the repository root
cmake -S deploy/camera -B deploy/camera/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build deploy/camera/build -j
```

Outputs:

```text
deploy/camera/build/d435i/d435i_depth_publisher
deploy/camera/build/viewer/depth_camera_viewer
```

## Usage

Build and start both programs:

```bash
./deploy/camera/run.sh --build
```

Start an existing build:

```bash
./deploy/camera/run.sh
```

The script displays both output streams. `Ctrl+C` or either process exiting shuts down the other process.

Start programs separately:

```bash
./deploy/camera/build/d435i/d435i_depth_publisher
./deploy/camera/build/viewer/depth_camera_viewer
```

Viewer controls:

| Input | Action |
| --- | --- |
| Terminal `R` | Start or stop recording |
| Terminal `Q` | Quit |
| Window `R` | Start or stop recording |
| Window `Esc` | Quit |
| DDS `ON` | Start recording |
| DDS `OFF` | Stop recording |
| Go2 / Unitree MuJoCo `Select+A` | Start or stop recording |

Recordings are written to:

```text
deploy/camera/videos/depth_YYYY-MM-DD_HH-MM-SS.mkv
```

## Configuration

D435i configuration:

```text
deploy/camera/d435i/config/config.yaml
```

Viewer configuration:

```text
deploy/camera/viewer/config/config.yaml
```

The DDS `domain_id` and `interface` values must match in both files. Local PC testing uses `lo` by default; select the actual network interface for robot deployment.

Default depth path:

```text
D435i 640x360 Z16 @ 30 Hz
-> metric float32
-> rt/front_depth_observation 32FC1 @ 15 Hz
-> Viewer grayscale preview / H.265 recording
```

## Implementation

- The publisher handles camera capture and DDS publishing.
- The viewer consumes the latest DDS depth frame and maps `0-2 m` to 8-bit grayscale.
- FFmpeg uses `libx265`, `preset=fast`, `CRF=26`, and an MKV container.
- `cpp_viewer/record_switch` uses `std_msgs::msg::dds_::String_` and accepts only `ON` and `OFF`.
- The viewer subscribes to `rt/wirelesscontroller` and detects the rising edge of `Select+A`.
- Recorded video is intended for visual debugging and does not preserve raw metric `32FC1` depth precision.
