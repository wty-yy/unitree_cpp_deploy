// Builds toward commands from the joystick for the flat-toward policy, which
// consumes [direction_x, direction_y, desired_speed]. The left stick selects an
// absolute world-frame heading (up = world +X, left = world +Y) and the right
// stick forward deflection gives the desired speed, matching
// deploy_go2_toward.py. The commanded direction is that heading expressed in the
// base frame, [cos(psi_w - yaw), sin(psi_w - yaw)], which a real robot builds
// from its heading estimate. A centered left stick keeps the last commanded
// heading so the robot turns in place instead of drifting.
//
// Enable it from the model's params/deploy.yaml:
//   commands:
//     base_velocity:
//       toward_command:
//         enabled: true
//         max_speed: 2.0
//         deadzone: 0.1
//         direction_angle_range: [-0.785398, 0.785398]  # rad, defaults to [-pi, pi]

#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

namespace utils
{

class TowardCommand
{
public:
    void configure(
        const YAML::Node& policy_cfg,
        unitree::common::UnitreeJoystick* joystick)
    {
        joystick_ = joystick;
        enabled_ = false;
        max_speed_ = 2.0f;
        deadzone_ = 0.1f;
        direction_angle_range_ = {-static_cast<float>(M_PI), static_cast<float>(M_PI)};

        const auto commands_cfg = policy_cfg["commands"];
        const auto base_velocity_cfg = commands_cfg
            ? commands_cfg["base_velocity"]
            : YAML::Node();
        const auto toward_cfg = base_velocity_cfg
            ? base_velocity_cfg["toward_command"]
            : YAML::Node();
        if (!toward_cfg || !toward_cfg["enabled"] || !toward_cfg["enabled"].as<bool>()) {
            return;
        }

        if (toward_cfg["max_speed"]) {
            max_speed_ = toward_cfg["max_speed"].as<float>();
            if (!std::isfinite(max_speed_) || max_speed_ < 0.0f) {
                throw std::invalid_argument(
                    "commands.base_velocity.toward_command.max_speed "
                    "must be finite and non-negative");
            }
        }
        if (toward_cfg["deadzone"]) {
            deadzone_ = toward_cfg["deadzone"].as<float>();
            if (!std::isfinite(deadzone_) || deadzone_ < 0.0f) {
                throw std::invalid_argument(
                    "commands.base_velocity.toward_command.deadzone "
                    "must be finite and non-negative");
            }
        }
        if (toward_cfg["direction_angle_range"]) {
            const auto angle_range =
                toward_cfg["direction_angle_range"].as<std::vector<float>>();
            if (angle_range.size() != 2) {
                throw std::invalid_argument(
                    "commands.base_velocity.toward_command.direction_angle_range "
                    "must contain two values");
            }
            if (!std::isfinite(angle_range[0]) || !std::isfinite(angle_range[1])) {
                throw std::invalid_argument(
                    "commands.base_velocity.toward_command.direction_angle_range "
                    "must be finite");
            }
            if (angle_range[0] > angle_range[1]) {
                throw std::invalid_argument(
                    "commands.base_velocity.toward_command.direction_angle_range "
                    "lower bound must not exceed upper bound");
            }
            direction_angle_range_ = {angle_range[0], angle_range[1]};
        }
        enabled_ = true;
        spdlog::info(
            "Toward command enabled: left stick world heading clamped to "
            "[{:.3f}, {:.3f}] rad, right stick forward speed up to {:.2f} m/s "
            "(deadzone {:.2f})",
            direction_angle_range_[0], direction_angle_range_[1],
            max_speed_, deadzone_);
    }

    bool enabled() const
    {
        return enabled_;
    }

    void update(
        bool use_fixed_command,
        const std::array<float, 3>& fixed_command,
        float root_heading_w)
    {
        if (use_fixed_command) {
            command_.assign(fixed_command.begin(), fixed_command.end());
            return;
        }

        if (joystick_ == nullptr) {
            throw std::runtime_error("TowardCommand joystick is not configured");
        }

        // Unitree remote: ly is forward-positive, lx is right-positive.
        const float forward = joystick_->ly();
        const float left = -joystick_->lx();
        if (std::hypot(forward, left) >= deadzone_) {
            world_heading_ = std::atan2(left, forward);
        }
        if (!world_heading_) {
            // Until the stick picks a bearing, hold the current heading.
            world_heading_ = root_heading_w;
        }

        float speed = joystick_->ry();
        if (std::fabs(speed) < deadzone_) {
            speed = 0.0f;
        }

        const float heading_error = std::clamp(
            wrap_to_pi(*world_heading_ - root_heading_w),
            direction_angle_range_[0], direction_angle_range_[1]);
        command_[0] = std::cos(heading_error);
        command_[1] = std::sin(heading_error);
        command_[2] = std::clamp(speed, 0.0f, 1.0f) * max_speed_;
    }

    void reset()
    {
        world_heading_.reset();
        command_.assign(3, 0.0f);
    }

    const std::vector<float>& command() const
    {
        return command_;
    }

private:
    static float wrap_to_pi(float angle)
    {
        return std::atan2(std::sin(angle), std::cos(angle));
    }

    bool enabled_ = false;
    float max_speed_ = 2.0f;
    float deadzone_ = 0.1f;
    std::array<float, 2> direction_angle_range_{
        -static_cast<float>(M_PI), static_cast<float>(M_PI)};
    unitree::common::UnitreeJoystick* joystick_ = nullptr;
    std::optional<float> world_heading_;
    std::vector<float> command_{0.0f, 0.0f, 0.0f};
};

} // namespace utils
