# Observation Group 配置说明

当前使用的**单 group (flat) 格式**, 所有 term 拼接为一个 vector 传入 ONNX 模型的唯一 input `"obs"`:
```yaml
# deploy.yaml — 单 group 格式
# ONNX 模型: 1个输入 input_name="obs", shape=[1, 490]
observations:
  use_gym_history: true
  base_ang_vel:
    params: {}
    scale: [1.0, 1.0, 1.0]
    clip: [-18.0, 18.0]
    history_length: 5
  joint_pos_rel:
    params: {}
    scale: [1.0, ...]
    clip: [-18.0, 18.0]
    history_length: 5
  # ... 其他 term
  # 所有 term 拼接 → {"obs": [490维 vector]} → 送入模型 input "obs"
```

如果模型有**多个输入**, 使用**多 group 格式**, 每个 group 名对应 ONNX 模型的一个 input name:
```yaml
# deploy.yaml — 多 group 格式
# ONNX 模型: 2个输入 input_name="policy" shape=[1,490], input_name="critic" shape=[1,187]
observations:
  policy:                    # ← group 名 = ONNX input name "policy"
    use_gym_history: true
    base_ang_vel:
      params: {}
      scale: [1.0, 1.0, 1.0]
      clip: [-18.0, 18.0]
      history_length: 5
    joint_pos_rel:
      params: {}
      scale: [1.0, ...]
      clip: [-18.0, 18.0]
      history_length: 5
    # → {"policy": [490维 vector]}
  critic:                    # ← group 名 = ONNX input name "critic"
    height_scan:
      params: {}
      scale: [1.0, ...]
      clip: [-5.0, 5.0]
      history_length: 1
    # → {"critic": [187维 vector]}
  # compute() 返回 {"policy": [...], "critic": [...]}
  # OrtRunner 按 ONNX input name 匹配 → 分别传入对应 input tensor
```
区分规则: 程序检查第一个非特殊 key 的 value 是否含 `params` 字段, 有则为 flat 单 group (自动归入 group `"obs"`), 无则为多 group 格式。因此 **flat 格式下每个 term 建议都写 `params: {}`**。