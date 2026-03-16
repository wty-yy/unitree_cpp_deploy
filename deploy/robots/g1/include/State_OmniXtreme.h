// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "onnxruntime_cxx_api.h"
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <random>
#include <mutex>
#include <thread>

namespace cnpy
{
struct NpyArray;
}

class State_OmniXtreme : public FSMState
{
public:
    State_OmniXtreme(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

private:
    struct KeyConfig
    {
        std::string next_trajectory = "Y.on_pressed";
        std::string previous_trajectory = "A.on_pressed";
        std::string reset_trajectory = "X.on_pressed";
        std::string toggle_execute = "B.on_pressed";
    };

    struct MotionTrajectory
    {
        std::string name;
        std::vector<std::vector<float>> ref_joint_pos_frames;
        std::vector<std::vector<float>> ref_joint_vel_frames;
        std::vector<Eigen::Quaternionf> ref_anchor_quat_frames;
        std::vector<Eigen::Quaternionf> ref_root_quat_frames;
    };

    void load_policy_and_env(const YAML::Node& cfg);
    void load_motion_library(const YAML::Node& cfg);
    void load_key_config(const YAML::Node& cfg);
    void initialize_limits(const YAML::Node& cfg);
    void calibrate_yaw_alignment();

    std::vector<float> build_real_obs(const std::vector<float>& motion_command) const;
    std::vector<float> build_command_obs(const MotionTrajectory& traj, std::size_t frame_index);
    std::vector<float> build_residual_obs(
        const std::vector<float>& real_obs,
        const std::vector<float>& command_obs,
        const std::vector<float>& base_action) const;
    std::vector<float> build_history_obs(const std::vector<float>& current_real_obs);
    std::vector<float> infer_base_action(
        const std::vector<float>& real_obs,
        const std::vector<float>& command_obs,
        const std::vector<float>& real_hist);
    std::vector<float> infer_residual_action(const std::vector<float>& residual_obs);
    std::vector<float> compute_q_target(const std::vector<float>& action) const;

    void reset_tracking_state(bool keep_current_traj);
    void handle_gamepad_events();
    void advance_frame();
    std::size_t paused_frame_index() const;

    static std::vector<float> to_float_vector(const cnpy::NpyArray& array);

private:
    YAML::Node deploy_cfg_;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;

    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "omnixtreme_onnx"};
    Ort::SessionOptions base_session_options_;
    Ort::SessionOptions residual_session_options_;
    std::unique_ptr<Ort::Session> base_session_;
    std::unique_ptr<Ort::Session> residual_session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<std::string> base_input_names_str_;
    std::vector<const char*> base_input_names_;
    std::vector<std::vector<int64_t>> base_input_shapes_;
    std::vector<std::string> base_output_names_str_;
    std::vector<const char*> base_output_names_;
    std::string base_action_output_name_ = "action";

    std::string residual_input_name_;
    std::vector<std::string> residual_output_names_str_;
    std::vector<const char*> residual_output_names_;
    std::string residual_action_output_name_ = "actions";

    std::unique_ptr<Ort::Session> fk_session_;
    std::vector<std::string> fk_output_names_str_;
    std::vector<const char*> fk_output_names_;

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};
    std::atomic<bool> execute_motion_{false};

    mutable std::mutex target_mtx_;
    std::vector<float> latest_q_target_;
    std::vector<float> latest_tau_ff_;

    std::function<bool(const unitree::common::UnitreeJoystick&)> next_trajectory_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> previous_trajectory_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> reset_trajectory_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> toggle_execute_trigger_;

    KeyConfig key_cfg_;
    std::vector<MotionTrajectory> trajectories_;
    std::size_t trajectory_index_{0};
    std::size_t frame_index_{0};
    std::size_t history_length_{15};
    std::deque<std::vector<float>> real_obs_history_;

    std::size_t dof_{0};
    std::size_t real_obs_dim_{90};
    std::size_t command_obs_dim_{64};
    std::size_t history_obs_dim_{1350};
    std::size_t residual_obs_dim_{183};
    int root_body_index_{0};
    int anchor_body_index_{9};

    float action_clip_{1.0f};
    float residual_scale_{1.0f};
    bool loop_trajectory_{true};
    std::size_t total_steps_{0};

    std::vector<float> action_scale_;
    std::vector<float> obs_default_joint_pos_;
    std::vector<float> joint_pos_lower_limit_;
    std::vector<float> joint_pos_upper_limit_;
    std::vector<float> p_gains_;
    std::vector<float> d_gains_;
    std::vector<float> x1_;
    std::vector<float> x2_;
    std::vector<float> y1_;
    std::vector<float> y2_;
    std::vector<float> va_;
    std::vector<float> fs_;
    std::vector<float> fd_;
    std::vector<float> last_action_;
    std::vector<float> last_base_action_;
    float initial_yaw_offset_{0.0f};
    mutable std::mt19937 rng_{std::random_device{}()};
    mutable std::normal_distribution<float> normal_dist_{0.0f, 1.0f};
};

REGISTER_FSM(State_OmniXtreme)
