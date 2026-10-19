// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include "utils/VelocityCommandDamper.h"
#include "utils/ExternalVelocity.h"
#include <iostream>
#include <map>
#include <string>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(
        YAML::Node policy_cfg,
        std::shared_ptr<Articulation> robot_,
        const YAML::Node& fsm_cfg = YAML::Node())
    :policy_cfg(policy_cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = policy_cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = policy_cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            std::vector<float> default_joint_pos(robot->data.joint_ids_map.size(), 0.0f);
            if (policy_cfg["default_joint_pos"])
            {
                default_joint_pos = policy_cfg["default_joint_pos"].as<std::vector<float>>();
            }
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness.assign(robot->data.joint_ids_map.size(), 0.0f);
            robot->data.joint_damping.assign(robot->data.joint_ids_map.size(), 0.0f);
            if (policy_cfg["stiffness"])
            {
                robot->data.joint_stiffness = policy_cfg["stiffness"].as<std::vector<float>>();
            }
            if (policy_cfg["damping"])
            {
                robot->data.joint_damping = policy_cfg["damping"].as<std::vector<float>>();
            }
        }

        robot->update();
        velocity_command_damper_.configure(this->policy_cfg, fsm_cfg, robot->data.joystick);

        // load managers
        if (policy_cfg["actions"])
        {
            action_manager = std::make_unique<ActionManager>(policy_cfg["actions"], this);
        }
        if (policy_cfg["observations"])
        {
            observation_manager = std::make_unique<ObservationManager>(policy_cfg["observations"], this);
        }
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        velocity_command_damper_.reset();
        velocity_command_damper_.update(
            step_dt,
            fixed_command_enabled && fixed_command_active,
            {fixed_lin_vel_x, fixed_lin_vel_y, fixed_ang_vel_z});
        update_external_velocity();
        if (action_manager) action_manager->reset();
        if (observation_manager) observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        velocity_command_damper_.update(
            step_dt,
            fixed_command_enabled && fixed_command_active,
            {fixed_lin_vel_x, fixed_lin_vel_y, fixed_ang_vel_z});
        update_external_velocity();
        if (!observation_manager || !action_manager || !alg)
        {
            throw std::runtime_error("ManagerBasedRLEnv::step requires observation_manager, action_manager and alg");
        }
        auto obs = observation_manager->compute();
        
        last_inference_results = alg->forward(obs);
        
        std::vector<float> action;
        if (last_inference_results.count("actions")) {
            action = last_inference_results["actions"];
        } else if (!last_inference_results.empty()) {
            action = last_inference_results.begin()->second;
        }
        
        action_manager->process_action(action);
    }

    const std::vector<float>& velocity_command() const
    {
        return velocity_command_damper_.command();
    }

    bool boost_active() const
    {
        return velocity_command_damper_.boost_active();
    }

    // Replace the joystick-derived command with the acknowledged external lease.
    void update_external_velocity()
    {
        auto& bridge = robot_automation::ExternalVelocity::instance();
        if (!external_velocity_enabled)
        {
            velocity_command_damper_.clear_external_command();
            return;
        }
        if (!velocity_command_damper_.has_command_ranges())
        {
            bridge.invalidate_ranges();
            velocity_command_damper_.clear_external_command();
            return;
        }
        const auto& active = velocity_command_damper_.active_ranges();
        robot_automation::ExternalVelocity::Ranges ranges;
        for (std::size_t i = 0; i < ranges.size(); ++i)
            ranges[i] = {active[i][0], active[i][1]};
        if (!bridge.set_ranges(ranges, velocity_command_damper_.boost_active()))
        {
            velocity_command_damper_.clear_external_command();
            return;
        }
        std::array<float, 3> command{};
        if (bridge.sample(command)) velocity_command_damper_.set_external_command(command);
        else velocity_command_damper_.clear_external_command();
    }

    float step_dt;
    
    YAML::Node policy_cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
    
    std::map<std::string, std::vector<float>> last_inference_results;

    // Set only by a Go2 Velocity_* state; the lease then replaces joystick commands.
    bool external_velocity_enabled = false;

    // Fixed command control
    bool fixed_command_enabled = false;
    bool fixed_command_active = false;
    float fixed_lin_vel_x = 0.0f;
    float fixed_lin_vel_y = 0.0f;
    float fixed_ang_vel_z = 0.0f;
    float fixed_command_duration = 0.0f;  // 0 means indefinite
    std::chrono::steady_clock::time_point fixed_command_start_time;

private:
    utils::VelocityCommandDamper velocity_command_damper_;
};

};
