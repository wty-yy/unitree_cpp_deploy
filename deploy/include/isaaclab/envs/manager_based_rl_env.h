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
#include <string>

namespace isaaclab
{

class ObservationManager;
class ActionManager;

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
            auto default_joint_pos = cfg["default_joint_pos"].as<std::vector<float>>();
            robot->data.default_joint_pos = Eigen::VectorXf::Map(default_joint_pos.data(), default_joint_pos.size());
        }
        { // joint stiffness and damping
            robot->data.joint_stiffness = cfg["stiffness"].as<std::vector<float>>();
            robot->data.joint_damping = cfg["damping"].as<std::vector<float>>();
        }

        robot->update();

        // load managers
        action_manager = std::make_unique<ActionManager>(cfg["actions"], this);
        observation_manager = std::make_unique<ObservationManager>(cfg["observations"], this);
    }

    void reset()
    {
        global_phase = 0;
        episode_length = 0;
        robot->update();
        if(robot->data.motion_loader) {
            robot->data.motion_loader->reset(robot->data);
        }
        action_manager->reset();
        observation_manager->reset();
    }

    void step()
    {
        episode_length += 1;
        robot->update();
        if(robot->data.motion_loader) {
            robot->data.motion_loader->update(episode_length * step_dt);
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
};

};