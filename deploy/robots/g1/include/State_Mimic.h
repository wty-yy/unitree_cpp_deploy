#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include <array>
#include <atomic>
#include <thread>

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

    class MotionLoader_;

    static std::shared_ptr<MotionLoader_> motion;

private:
    void reset_motion_state();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;
    std::shared_ptr<MotionLoader_> motion_;

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};
    std::atomic<bool> execute_motion_{false};
    std::atomic<bool> motion_finished_{false};

    std::array<float, 2> time_range_{};
    std::atomic<float> reference_time_{0.0f};
    int finished_state_id_{0};
};

class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(std::string motion_file, float fps);

    void update(float time);
    void reset(const isaaclab::ArticulationData& data, float t = 0.0f);

    Eigen::VectorXf joint_pos();
    Eigen::VectorXf root_position();
    Eigen::VectorXf joint_vel();
    Eigen::Quaternionf root_quaternion();

    float dt;
    int num_frames;
    float duration;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    Eigen::Matrix3f world_to_init_;

private:
    int index_0_{0};
    int index_1_{0};
    float blend_{0.0f};
};

REGISTER_FSM(State_Mimic)
