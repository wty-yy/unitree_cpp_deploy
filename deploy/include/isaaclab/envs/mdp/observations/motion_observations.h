// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_joint_pos)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto & ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_pos();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for(int i = 0; i < data_dfs.size(); ++i) {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_joint_vel)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto & ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_vel();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for(int i = 0; i < data_dfs.size(); ++i) {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_command)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto & ids = robot->data.joint_ids_map;

    auto pos_dfs = loader->joint_pos();
    Eigen::VectorXf pos_bfs = Eigen::VectorXf::Zero(pos_dfs.size());
    for(int i = 0; i < pos_dfs.size(); ++i) {
        pos_bfs(i) = pos_dfs[ids[i]];
    }
    auto vel_dfs = loader->joint_vel();
    Eigen::VectorXf vel_bfs = Eigen::VectorXf::Zero(vel_dfs.size());
    for(int i = 0; i < vel_dfs.size(); ++i) {
        vel_bfs(i) = vel_dfs[ids[i]];
    }
    std::vector<float> data;
    data.insert(data.end(), pos_bfs.data(), pos_bfs.data() + pos_bfs.size());
    data.insert(data.end(), vel_bfs.data(), vel_bfs.data() + vel_bfs.size());
    return data;
}

} // namespace mdp
} // namespace isaaclab
