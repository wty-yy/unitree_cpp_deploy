# Changelog

## v0.1 - 2026-11-09

1. Add Skill-MoE loading for `history` and `height_map` inputs, resolve dynamic batch to `1`, and validate tensor shapes and types
2. Expose initialized FSM key bindings and resolved policy directories through the read-only `policy_catalog` bridge query
3. Accept shifted height-map origins and require matching timestamps, widths, and origins for elevation and missing-mask pairing
4. Add standalone CMake bridge tests for Go2 four-direction bindings and read-only catalog queries
5. Set `Velocity_Down` to `skill_moe/mixed_climb_5k`, set `Velocity_Left` to `skill_moe/mixed_stairs_5k`, and disable `Velocity_Right`
