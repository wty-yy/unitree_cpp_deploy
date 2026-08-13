// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

// Builds velocity commands from joystick or fixed-command inputs and applies
// configurable speed-dependent first-order damping and acceleration limits to
// vx. Lateral and yaw commands remain unfiltered.

#pragma once

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils
{

class VelocityCommandDamper
{
public:
    void configure(
        const YAML::Node& policy_cfg,
        const YAML::Node& fsm_cfg,
        unitree::common::UnitreeJoystick* joystick)
    {
        joystick_ = joystick;
        configure_command_ranges(policy_cfg);
        configure_damping(fsm_cfg);
    }

    void update(
        float dt,
        bool use_fixed_command,
        const std::array<float, 3>& fixed_command)
    {
        if (!has_command_ranges_) {
            command_.assign(3, 0.0f);
            return;
        }

        if (use_fixed_command) {
            std::copy(fixed_command.begin(), fixed_command.end(), command_.begin());
        } else {
            if (joystick_ == nullptr) {
                throw std::runtime_error("VelocityCommandDamper joystick is not configured");
            }

            command_[0] = scale_command(joystick_->ly(), command_ranges_[0]);
            command_[1] = scale_command(-joystick_->lx(), command_ranges_[1]);
            command_[2] = scale_command(-joystick_->rx(), command_ranges_[2]);
        }

        command_[0] = update_vx(command_[0], dt);
    }

    void reset()
    {
        vx_ = 0.0f;
        command_.assign(3, 0.0f);
    }

    const std::vector<float>& command() const
    {
        return command_;
    }

private:
    using Range = std::array<float, 2>;

    void configure_command_ranges(const YAML::Node& policy_cfg)
    {
        has_command_ranges_ = false;

        const auto commands_cfg = policy_cfg["commands"];
        if (!commands_cfg || !commands_cfg["base_velocity"]) {
            return;
        }

        const auto base_velocity_cfg = commands_cfg["base_velocity"];
        if (!base_velocity_cfg["ranges"]) {
            return;
        }

        const auto ranges_cfg = base_velocity_cfg["ranges"];
        command_ranges_[0] = parse_range(ranges_cfg, "lin_vel_x");
        command_ranges_[1] = parse_range(ranges_cfg, "lin_vel_y");
        command_ranges_[2] = parse_range(ranges_cfg, "ang_vel_z");
        has_command_ranges_ = true;
    }

    void configure_damping(const YAML::Node& fsm_cfg)
    {
        enabled_ = false;
        max_acceleration_ = std::numeric_limits<float>::infinity();
        max_deceleration_ = std::numeric_limits<float>::infinity();
        speed_points_.clear();
        time_constants_.clear();

        const auto damping_cfg = fsm_cfg ? fsm_cfg["velocity_command_damping"] : YAML::Node();
        enabled_ = damping_cfg && damping_cfg["enabled"]
            ? damping_cfg["enabled"].as<bool>()
            : false;
        if (!enabled_) {
            return;
        }

        const auto vx_cfg = damping_cfg["vx"];
        if (!vx_cfg || !vx_cfg["speed_points"] || !vx_cfg["time_constants"]) {
            throw std::invalid_argument(
                "velocity_command_damping.vx requires speed_points and time_constants");
        }

        const auto speed_points = vx_cfg["speed_points"].as<std::vector<float>>();
        const auto time_constants = vx_cfg["time_constants"].as<std::vector<float>>();
        max_acceleration_ = parse_rate_limit(vx_cfg, "max_acceleration");
        max_deceleration_ = parse_rate_limit(vx_cfg, "max_deceleration");
        if (speed_points.size() < 2 || speed_points.size() != time_constants.size()) {
            throw std::invalid_argument(
                "VelocityCommandDamper requires equally sized speed_points and time_constants with at least two entries");
        }

        for (std::size_t i = 0; i < speed_points.size(); ++i) {
            if (!std::isfinite(speed_points[i]) || speed_points[i] < 0.0f) {
                throw std::invalid_argument("VelocityCommandDamper speed_points must be finite and non-negative");
            }
            if (!std::isfinite(time_constants[i]) || time_constants[i] < 0.0f) {
                throw std::invalid_argument("VelocityCommandDamper time_constants must be finite and non-negative");
            }
            if (i > 0 && speed_points[i] <= speed_points[i - 1]) {
                throw std::invalid_argument("VelocityCommandDamper speed_points must be strictly increasing");
            }
        }

        speed_points_ = speed_points;
        time_constants_ = time_constants;
    }

    static Range parse_range(const YAML::Node& ranges_cfg, const char* key)
    {
        if (!ranges_cfg[key]) {
            throw std::invalid_argument(std::string("Missing velocity command range: ") + key);
        }

        const auto values = ranges_cfg[key].as<std::vector<float>>();
        if (values.size() != 2) {
            throw std::invalid_argument(std::string("Velocity command range must contain two values: ") + key);
        }
        return {values[0], values[1]};
    }

    static float parse_rate_limit(const YAML::Node& vx_cfg, const char* key)
    {
        if (!vx_cfg[key]) {
            return std::numeric_limits<float>::infinity();
        }

        const float value = vx_cfg[key].as<float>();
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::invalid_argument(std::string(key) + " must be finite and positive");
        }
        return value;
    }

    static float scale_command(float command, const Range& range)
    {
        return command > 0.0f
            ? command * range[1]
            : command * -range[0];
    }

    float update_vx(float target, float dt)
    {
        if (!std::isfinite(target)) {
            throw std::invalid_argument("VelocityCommandDamper target must be finite");
        }

        if (!enabled_) {
            vx_ = target;
            return vx_;
        }
        if (!std::isfinite(dt) || dt <= 0.0f) {
            throw std::invalid_argument("VelocityCommandDamper dt must be finite and positive");
        }

        const float reference_speed = std::max(std::fabs(vx_), std::fabs(target));
        const float time_constant = interpolate_time_constant(reference_speed);
        float damped_target = target;
        if (time_constant > 0.0f) {
            const float alpha = 1.0f - std::exp(-dt / time_constant);
            damped_target = vx_ + alpha * (target - vx_);
        }

        const bool has_rate_limit = std::isfinite(max_acceleration_)
            || std::isfinite(max_deceleration_);
        if (has_rate_limit && vx_ * damped_target < 0.0f) {
            damped_target = 0.0f;
        }

        const float delta = damped_target - vx_;
        const bool decelerating = vx_ * delta < 0.0f;
        const float rate_limit = decelerating ? max_deceleration_ : max_acceleration_;
        const float max_delta = rate_limit * dt;
        vx_ += std::clamp(delta, -max_delta, max_delta);
        return vx_;
    }

    float interpolate_time_constant(float speed) const
    {
        if (speed <= speed_points_.front()) {
            return time_constants_.front();
        }
        if (speed >= speed_points_.back()) {
            return time_constants_.back();
        }

        const auto upper = std::upper_bound(speed_points_.begin(), speed_points_.end(), speed);
        const std::size_t upper_index = std::distance(speed_points_.begin(), upper);
        const std::size_t lower_index = upper_index - 1;
        const float span = speed_points_[upper_index] - speed_points_[lower_index];
        const float ratio = (speed - speed_points_[lower_index]) / span;
        return time_constants_[lower_index]
            + ratio * (time_constants_[upper_index] - time_constants_[lower_index]);
    }

    bool enabled_ = false;
    bool has_command_ranges_ = false;
    float vx_ = 0.0f;
    float max_acceleration_ = std::numeric_limits<float>::infinity();
    float max_deceleration_ = std::numeric_limits<float>::infinity();
    unitree::common::UnitreeJoystick* joystick_ = nullptr;
    std::array<Range, 3> command_ranges_{};
    std::vector<float> command_{0.0f, 0.0f, 0.0f};
    std::vector<float> speed_points_;
    std::vector<float> time_constants_;
};

} // namespace utils
