// Builds toward commands from the joystick for the climb-toward policy, which
// consumes [direction_x, direction_y, desired_speed]. The left stick gives a
// unit direction in the body frame and the right stick forward deflection
// gives the desired speed, matching deploy_go2_depth_toward.py.
//
// Enable it from the model's params/deploy.yaml:
//   commands:
//     base_velocity:
//       toward_command:
//         enabled: true
//         max_speed: 1.0
//         deadzone: 0.1
//         direction_angle_range: [-0.785398, 0.785398]  # rad, defaults to [-pi, pi]

#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
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
        max_speed_ = 1.0f;
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
            "Toward command enabled: left stick unit direction clamped to "
            "[{:.3f}, {:.3f}] rad, right stick forward speed up to {:.2f} m/s "
            "(deadzone {:.2f})",
            direction_angle_range_[0], direction_angle_range_[1],
            max_speed_, deadzone_);
    }

    bool enabled() const
    {
        return enabled_;
    }

    void update(bool use_fixed_command, const std::array<float, 3>& fixed_command)
    {
        if (use_fixed_command) {
            command_.assign(fixed_command.begin(), fixed_command.end());
            return;
        }

        if (joystick_ == nullptr) {
            throw std::runtime_error("TowardCommand joystick is not configured");
        }

        float direction_x = joystick_->ly();
        float direction_y = -joystick_->lx();
        if (std::fabs(direction_x) < deadzone_) {
            direction_x = 0.0f;
        }
        if (std::fabs(direction_y) < deadzone_) {
            direction_y = 0.0f;
        }

        const float direction_norm = std::sqrt(
            direction_x * direction_x + direction_y * direction_y);
        if (direction_norm < 1.0e-6f) {
            // No direction: hold still rather than emit an undefined bearing.
            command_.assign(3, 0.0f);
            return;
        }

        float forward = joystick_->ry();
        if (std::fabs(forward) < deadzone_) {
            forward = 0.0f;
        }
        const float angle = std::clamp(
            std::atan2(direction_y, direction_x),
            direction_angle_range_[0], direction_angle_range_[1]);
        command_[0] = std::cos(angle);
        command_[1] = std::sin(angle);
        command_[2] = std::clamp(forward, 0.0f, 1.0f) * max_speed_;
    }

    void reset()
    {
        command_.assign(3, 0.0f);
    }

    const std::vector<float>& command() const
    {
        return command_;
    }

private:
    bool enabled_ = false;
    float max_speed_ = 1.0f;
    float deadzone_ = 0.1f;
    std::array<float, 2> direction_angle_range_{
        -static_cast<float>(M_PI), static_cast<float>(M_PI)};
    unitree::common::UnitreeJoystick* joystick_ = nullptr;
    std::vector<float> command_{0.0f, 0.0f, 0.0f};
};

} // namespace utils
