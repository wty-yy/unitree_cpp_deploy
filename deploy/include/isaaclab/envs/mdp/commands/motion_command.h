#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include "isaaclab/assets/articulation/articulation.h"

namespace isaaclab
{

inline Eigen::Quaternionf yawQuaternion(const Eigen::Quaternionf& q) {
  float yaw = std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()), 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
  float half_yaw = yaw * 0.5f;
  Eigen::Quaternionf ret(std::cos(half_yaw), 0.0f, 0.0f, std::sin(half_yaw));
  return ret.normalized();
};

/* Unitree CSV Motion File */
class MotionLoader
{
public:
    MotionLoader(std::string motion_file, float fps = 50.0f)
    : dt(1.0f / fps)
    {
        auto data = _read_csv(motion_file);
        
        num_frames = data.size();
        duration = num_frames * dt;
        
        for(int i(0); i < num_frames; ++i)
        {
            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][6],data[i][3], data[i][4], data[i][5]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
        }
        dof_velocities = _comupte_raw_derivative(dof_positions);

        update(0.0f);
    }

    void update(float time) 
    {
        float phase = std::clamp(time / duration, 0.0f, 1.0f);
        index_0_ = std::round(phase * (num_frames - 1));
        index_1_ = std::min(index_0_ + 1, num_frames - 1);
        blend_ = std::round((time - index_0_ * dt) / dt * 1e5f) / 1e5f;
    }

    void reset(const ArticulationData & data)
    {
        update(0.0f);
        auto init_to_anchor = yawQuaternion(this->root_quaternion()).toRotationMatrix();
        auto world_to_anchor = yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf joint_pos() {
        return dof_positions[index_0_] * (1 - blend_) + dof_positions[index_1_] * blend_;
    }

    Eigen::VectorXf root_position() {
        return root_positions[index_0_] * (1 - blend_) + root_positions[index_1_] * blend_;
    }

    Eigen::VectorXf joint_vel() {
        return dof_velocities[index_0_] * (1 - blend_) + dof_velocities[index_1_] * blend_;
    }

    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]);
    }

    float dt;
    int num_frames;
    float duration;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    Eigen::Matrix3f world_to_init_;
private:
    int index_0_;
    int index_1_;
    float blend_;

    std::vector<Eigen::VectorXf> _comupte_raw_derivative(const std::vector<Eigen::VectorXf>& data)
    {
        std::vector<Eigen::VectorXf> derivative;
        for(int i = 0; i < data.size() - 1; ++i) {
            derivative.push_back((data[i + 1] - data[i]) / dt);
        }
        derivative.push_back(derivative.back());
        return derivative;
    }

    std::vector<std::vector<float>> _read_csv(const std::string& filename)
    {
        std::vector<std::vector<float>> data;
        std::ifstream file(filename);
        if (!file.is_open())
        {
            spdlog::error("Error opening file: {}", filename);
            return data;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ','))
            {
                try
                {
                    row.push_back(std::stof(value));
                }
                catch (const std::invalid_argument& e)
                {
                    spdlog::error("Invalid value in file: {}", value);
                }
            }
            data.push_back(row);
        }
        file.close();
        return data;
    }
};

};