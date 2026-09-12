# AGENTS.md

C++17 RL policy deployment for Unitree Go2 (12-DOF) and G1 (29-DOF). No tests, lint, or CI: verify changes by building the affected target.

## Layout

- `deploy/include/`: shared FSM + Isaac Lab-style observation/action framework
- `deploy/robots/go2/`, `deploy/robots/g1/`: controller sources, `main.cpp`, `config/config.yaml`
- `deploy/camera/`: standalone D435i depth publisher + viewer (OpenCV, RealSense, FFmpeg); see `deploy/camera/README.md`
- `deploy/thirdparty/`: bundled `cnpy`/`nlohmann`; `onnxruntime-linux-*` is gitignored and must be downloaded manually
- `logs/`: policy weights (gitignored directory; tracked models were force-added)
- `docs/`: `robot_params.md` (FSM/config DSL), `g1_setup_zh.md` (G1/BFM/OmniXtreme), `obs_group.md`

## Build

```bash
# Go2 -> deploy/robots/go2/build/go2_ctrl
cd deploy/robots/go2 && cmake -B build && cmake --build build -j$(nproc)
# G1 -> deploy/robots/g1/build/g1_ctrl (also builds bundled cnpy)
cd deploy/robots/g1 && cmake -B build && cmake --build build -j$(nproc)
# Camera tools
./deploy/camera/run.sh --build
```

- `deploy/robots/{go2,g1}/CMakeLists.txt` hardcode the `onnxruntime-linux-x64-gpu-1.24.2` path. For Orin NX, comment the x64 lines and uncomment the `aarch64-gpu-1.16.0` lines.
- Links against system-installed `unitree_sdk2`, ddsc/ddscxx, Boost `program_options`, yaml-cpp, Eigen, fmt; apt list in `README.md` / `docs/g1_setup_zh.md`.
- `CMAKE_EXPORT_COMPILE_COMMANDS` is on; `.vscode` points at the g1 build dir.

## Weights

- `*.onnx` / `*.pt` are Git LFS files: run `git lfs pull`; ~130-byte files are pointers, not corrupt models.
- RLBase policy dir contract: `{policy_dir}/exported/policy.onnx` + `{policy_dir}/params/deploy.yaml`; G1 BFM/OmniXtreme need extra files (see `docs/g1_setup_zh.md`).
- Relative `policy_dir` values in `config.yaml` resolve against the robot dir (parent of `build/`), not CWD. If a dir has no `exported/`, the newest subdir containing it is selected.
- `logs/` is gitignored: new weights need `git add -f`; LFS uploads the content automatically.

## FSM / config

- States are declared in `config.yaml` under `FSM._`; the first entry is the initial state; `type` selects the state class without the `State_` prefix (defaults to the state name).
- New state classes inherit a base state, add `REGISTER_FSM(Class)`, and use the `(int state, std::string state_string)` constructor.
- Transitions are per-state DSL strings evaluated by `deploy/include/unitree_joystick_dsl.hpp`; syntax in `docs/robot_params.md`.

## Conventions

- Do not strip Apache headers from Unitree-derived files; other source files carry no license header.
- Keep README pairs in sync: `README.md`/`README_zh.md` (root), `README.md`/`README.zh-CN.md` (camera).
- `CHANGELOG.md` uses `## vX.Y.Z YYYY-MM-DD` headings, newest first.
