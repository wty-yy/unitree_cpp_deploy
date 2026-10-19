// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "unitree_joystick_dsl.hpp"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils
{

// Model-specific velocity ranges, active while the configured button is held.
class VelocityCommandBoost
{
public:
    void configure(
        const YAML::Node& policy_cfg,
        const YAML::Node& fsm_cfg,
        unitree::common::UnitreeJoystick* joystick)
    {
        button_ = nullptr;
        ranges_.reset();

        const auto commands = policy_cfg["commands"];
        if (!commands || !commands["base_velocity"])
        {
            return;
        }
        const auto boost = commands["base_velocity"]["boost"];
        if (!boost || (boost["enabled"] && !boost["enabled"].as<bool>()))
        {
            return;
        }
        if (joystick == nullptr)
        {
            throw std::invalid_argument("Velocity command boost requires a joystick");
        }

        const auto ranges = boost["ranges"] ? boost["ranges"] : boost;
        for (const auto* key : {"lin_vel_x", "lin_vel_y", "ang_vel_z"})
        {
            if (!ranges[key])
            {
                throw std::invalid_argument(std::string("Missing boost velocity command range: ") + key);
            }
            const auto values = ranges[key].as<std::vector<float>>();
            if (values.size() != 2 || !std::isfinite(values[0]) || !std::isfinite(values[1]) ||
                values[0] > 0.0f || values[1] < 0.0f || values[0] > values[1])
            {
                throw std::invalid_argument(
                    std::string("Boost velocity command range must contain two finite values spanning zero: ") + key);
            }
        }

        const std::string button_name = fsm_cfg && fsm_cfg["boost_button"]
            ? fsm_cfg["boost_button"].as<std::string>()
            : "RT";
        button_ = &unitree::common::dsl::GetKey(*joystick, button_name);
        ranges_.reset(ranges);

        spdlog::info(
            "Velocity command boost enabled: hold [{}] for "
            "lin_vel_x=[{:.2f}, {:.2f}], lin_vel_y=[{:.2f}, {:.2f}], ang_vel_z=[{:.2f}, {:.2f}]",
            button_name,
            ranges_["lin_vel_x"][0].as<float>(), ranges_["lin_vel_x"][1].as<float>(),
            ranges_["lin_vel_y"][0].as<float>(), ranges_["lin_vel_y"][1].as<float>(),
            ranges_["ang_vel_z"][0].as<float>(), ranges_["ang_vel_z"][1].as<float>());
    }

    bool active() const
    {
        return button_ != nullptr && button_->pressed;
    }

    const YAML::Node& ranges() const
    {
        return ranges_;
    }

private:
    const unitree::common::KeyBase* button_ = nullptr;
    YAML::Node ranges_;
};

} // namespace utils
