// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "onnxruntime_cxx_api.h"
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>

namespace cnpy
{
struct NpyArray;
}

class State_BFM : public FSMState
{
public:
    State_BFM(int state_mode, std::string state_string);

    void enter();
    void run();
    void exit();

private:
    enum class TaskType
    {
        Tracking,
        Reward,
        Goal
    };

    struct KeyConfig
    {
        std::string start_motion = "B.on_pressed";
        std::string next_latent = "Y.on_pressed";
        std::string reset_state = "X.on_pressed";
    };

    void load_policy_and_env(const YAML::Node& cfg);
    void load_task_context(const YAML::Node& cfg);
    void load_key_config(const YAML::Node& cfg);
    void initialize_limits(const YAML::Node& cfg);

    std::vector<float> build_policy_obs() const;
    std::vector<float> build_policy_input();

    std::vector<float> compute_tracking_latent();
    std::vector<float> get_current_latent() const;
    std::vector<float> infer_action(const std::vector<float>& input);

    std::vector<float> compute_q_target(const std::vector<float>& scaled_action);
    void handle_gamepad_events();

    static std::vector<float> to_float_vector(const cnpy::NpyArray& array);

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;

    // ONNX runtime (BFM custom path; CUDA first, CPU fallback)
    Ort::Env ort_env_{ORT_LOGGING_LEVEL_WARNING, "bfm_onnx"};
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string onnx_input_name_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;
    std::string action_output_name_ = "actions";

    std::thread policy_thread_;
    std::atomic<bool> policy_thread_running_{false};

    mutable std::mutex target_mtx_;
    std::vector<float> latest_q_target_;

    bool enable_bad_orientation_check_{false};
    float bad_orientation_limit_{1.0f};
    float bad_orientation_grace_s_{1.0f};
    std::chrono::steady_clock::time_point enter_time_;

    std::function<bool(const unitree::common::UnitreeJoystick&)> start_motion_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> next_latent_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> reset_state_trigger_;

    // Task / latent state
    TaskType task_type_{TaskType::Goal};
    std::size_t latent_dim_{256};
    std::vector<std::vector<float>> selected_latents_;
    std::vector<std::string> selected_latent_names_;
    std::size_t z_index_{0};

    // tracking context (T x latent_dim)
    std::vector<std::vector<float>> tracking_ctx_;
    int t_{0};
    int t_start_{0};
    int t_end_{0};
    int t_stop_{0};
    float gamma_{0.8f};
    int window_size_{3};

    // control state
    KeyConfig key_cfg_;
    bool use_policy_action_{false};
    bool start_motion_{false};
    std::size_t total_steps_{0};

    float action_rescale_{5.0f};
    std::vector<float> action_scale_;
    std::vector<float> default_joint_pos_;
    std::vector<float> joint_pos_lower_limit_;
    std::vector<float> joint_pos_upper_limit_;
    std::vector<float> last_action_;

};

REGISTER_FSM(State_BFM)
