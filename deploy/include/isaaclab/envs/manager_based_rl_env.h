// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "isaaclab/manager/observation_manager.h"
#include "isaaclab/manager/action_manager.h"
#include "isaaclab/envs/mdp/commands/motion_command.h"
#include "isaaclab/assets/articulation/articulation.h"
#include "isaaclab/algorithms/algorithms.h"
#include <iostream>
#include <map>
#include <mutex>
#include <string>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

struct PolicyLoggingSnapshot
{
    std::vector<std::pair<std::string, std::vector<float>>> observation_groups;
    std::vector<ObservationTermSnapshot> observation_terms;
    std::map<std::string, std::vector<float>> inference_results;
    std::vector<float> raw_action;
    std::vector<float> processed_action;
};

class ManagerBasedRLEnv
{
public:
    // Constructor
    ManagerBasedRLEnv(YAML::Node cfg, std::shared_ptr<Articulation> robot_)
    :cfg(cfg), robot(std::move(robot_))
    {
        // Parse configuration
        this->step_dt = cfg["step_dt"].as<float>();
        robot->data.joint_ids_map = cfg["joint_ids_map"].as<std::vector<float>>();
        robot->data.joint_pos.resize(robot->data.joint_ids_map.size());
        robot->data.joint_vel.resize(robot->data.joint_ids_map.size());

        { // default joint positions
            std::vector<float> default_joint_pos(robot->data.joint_ids_map.size(), 0.0f);
            if (cfg["default_joint_pos"])
            {
                default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            }
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness.assign(robot->data.joint_ids_map.size(), 0.0f);
            robot->data.joint_damping.assign(robot->data.joint_ids_map.size(), 0.0f);
            if (cfg["stiffness"])
            {
                robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            }
            if (cfg["damping"])
            {
                robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
            }
        }

        robot->update();

        // load managers
        if (cfg["actions"])
        {
            action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        }
        if (cfg["observations"])
        {
            observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
        }
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        if(robot->data.motion_loader) {
            robot->data.motion_loader->reset(robot->data);
        }
        if (action_manager) action_manager->reset();
        if (observation_manager) observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        if(robot->data.motion_loader) {
            robot->data.motion_loader->update(episode_length * step_dt);
        }
        if (!observation_manager || !action_manager || !alg)
        {
            throw std::runtime_error("ManagerBasedRLEnv::step requires observation_manager, action_manager and alg");
        }
        auto obs = observation_manager->compute();
        auto inference_results = alg->forward(obs);

        std::vector<float> action;
        if (inference_results.count("actions")) {
            action = inference_results["actions"];
        } else if (!inference_results.empty()) {
            action = inference_results.begin()->second;
        }

        action_manager->process_action(action);
        const auto processed_action = action_manager->processed_actions();

        {
            std::lock_guard<std::mutex> lock(policy_snapshot_mutex_);
            last_policy_snapshot_.observation_groups = observation_manager->last_group_observations();
            last_policy_snapshot_.observation_terms = observation_manager->last_term_observations();
            last_policy_snapshot_.inference_results = inference_results;
            last_policy_snapshot_.raw_action = action;
            last_policy_snapshot_.processed_action = processed_action;
            last_inference_results = inference_results;
        }
    }

    PolicyLoggingSnapshot get_policy_logging_snapshot() const
    {
        std::lock_guard<std::mutex> lock(policy_snapshot_mutex_);
        return last_policy_snapshot_;
    }

    float step_dt;
    
    YAML::Node cfg;

    std::unique_ptr<ObservationManager> observation_manager;
    std::unique_ptr<ActionManager> action_manager;
    std::shared_ptr<Articulation> robot;
    std::unique_ptr<Algorithms> alg;
    long episode_length = 0;
    float global_phase = 0.0f;
    
    std::map<std::string, std::vector<float>> last_inference_results;

    // Fixed command control
    bool fixed_command_enabled = false;
    bool fixed_command_active = false;
    float fixed_lin_vel_x = 0.0f;
    float fixed_lin_vel_y = 0.0f;
    float fixed_ang_vel_z = 0.0f;
    float fixed_command_duration = 0.0f;  // 0 means indefinite
    std::chrono::steady_clock::time_point fixed_command_start_time;

private:
    mutable std::mutex policy_snapshot_mutex_;
    PolicyLoggingSnapshot last_policy_snapshot_;
};

};
