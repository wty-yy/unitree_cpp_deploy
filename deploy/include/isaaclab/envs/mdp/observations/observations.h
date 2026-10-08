// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once
#include "utils/ExternalVelocity.h"

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    if(params["asset_cfg"]["joint_ids"])
    {
        auto joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }
    else
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    if(params["asset_cfg"]["joint_ids"])
    {
        auto joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i) {
            data[i] = asset->data.joint_pos[joint_ids[i]] - asset->data.default_joint_pos[joint_ids[i]];
        }
    }
    else
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
            data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;

    if(params["asset_cfg"]["joint_ids"])
    {
        auto joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        std::vector<float> data(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i) {
            data[i] = asset->data.joint_vel[joint_ids[i]];
        }
        return data;
    }

    auto & data = asset->data.joint_vel;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    YAML::Node cfg;
    if (env->external_velocity_enabled)
    {
        auto& bridge = robot_automation::ExternalVelocity::instance();
        robot_automation::ExternalVelocity::Ranges ranges;
        try {
            cfg = env->velocity_command_ranges();
            const std::array<const char*, 3> keys{"lin_vel_x", "lin_vel_y", "ang_vel_z"};
            for (std::size_t i = 0; i < keys.size(); ++i) {
                const auto axis = cfg[keys[i]];
                if (!axis.IsSequence() || axis.size() != 2) {
                    bridge.invalidate_ranges();
                    return std::vector<float>(3, 0.0f);
                }
                ranges[i] = {axis[0].as<double>(), axis[1].as<double>()};
            }
        } catch (const YAML::Exception&) {
            bridge.invalidate_ranges();
            return std::vector<float>(3, 0.0f);
        }
        if (!bridge.set_ranges(ranges, env->boost_active()))
            return std::vector<float>(3, 0.0f);
        std::array<float, 3> external;
        if (bridge.sample(external))
            return std::vector<float>(external.begin(), external.end());
    }
    else
    {
        cfg = env->velocity_command_ranges();
    }

    std::vector<float> obs(3);

    // Check if fixed command mode is active
    if (env->fixed_command_enabled && env->fixed_command_active) {
        obs[0] = env->fixed_lin_vel_x;
        obs[1] = env->fixed_lin_vel_y;
        obs[2] = env->fixed_ang_vel_z;
        return obs;
    }

    auto & joystick = env->robot->data.joystick;

    obs[0] = joystick->ly();
    obs[1] = -joystick->lx();
    obs[2] = -joystick->rx();

    auto scale_func = [&cfg](std::vector<float>& obs, int idx, const std::string& key) {
        if (obs[idx] > 0) {
            obs[idx] *= cfg[key][1].as<float>();
        } else {
            obs[idx] *= -cfg[key][0].as<float>();
        }
    };
    scale_func(obs, 0, "lin_vel_x");
    scale_func(obs, 1, "lin_vel_y");
    scale_func(obs, 2, "ang_vel_z");

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}

}
}
