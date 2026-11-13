// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "DataLogger.h"
#include "utils/filters/joint_vector_filter.h"
#include "utils/G1LoggingTelemetry.h"
#include <atomic>
#include <chrono>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    static FsmPreflightResult preflight(const YAML::Node& cfg, const std::string& state_name);
    
    void enter()
    {
        if (!env) {
            spdlog::warn("State_RLBase::enter: Environment is null. Skipping initialization.");
            return;
        }

        const auto enter_start = std::chrono::steady_clock::now();
        start_logging();

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        filtered_action_.assign(
            env->robot->data.joint_pos.data(),
            env->robot->data.joint_pos.data() + env->robot->data.joint_pos.size());
        action_filter_.reset(filtered_action_);
        entry_inference_seq_ = env->get_policy_logging_snapshot().inference_seq;
        for (std::size_t i = 0; i < filtered_action_.size(); ++i)
        {
            const auto joint_id = static_cast<int>(env->robot->data.joint_ids_map[i]);
            lowcmd->msg_.motor_cmd()[joint_id].q() = filtered_action_[i];
        }
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this, enter_start]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            const auto reset_start = std::chrono::steady_clock::now();
            // A deferred switch already ran this policy's frames; keep its history.
            if (!prewarmed_.exchange(false))
            {
                env->reset();
            }
            const auto reset_done = std::chrono::steady_clock::now();

            bool first_step = true;
            while (policy_thread_running)
            {
                env->step();
                if (first_step)
                {
                    first_step = false;
                    const auto step_done = std::chrono::steady_clock::now();
                    spdlog::info(
                        "State_{}: policy ready in {:.1f} ms (enter {:.1f} ms, reset {:.1f} ms, first step {:.1f} ms)",
                        getStateString(),
                        std::chrono::duration<double, std::milli>(step_done - enter_start).count(),
                        std::chrono::duration<double, std::milli>(reset_start - enter_start).count(),
                        std::chrono::duration<double, std::milli>(reset_done - reset_start).count(),
                        std::chrono::duration<double, std::milli>(step_done - reset_done).count());
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });

        spdlog::info("State_{}: enter took {:.1f} ms", getStateString(),
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - enter_start).count());
    }

    void run();
    void warmup() override;
    bool start_policy_prewarm(int frames) override;
    bool policy_prewarm_done() const override { return prewarm_done_.load(); }
    void finish_policy_prewarm() override;

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        if (prewarm_thread_.joinable()) {
            prewarm_thread_.join();
        }
    }

private:
    void start_logging();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<DataLogger> logger;
    bool enable_logging = false;
    std::chrono::duration<double> logging_dt{0.02};
    bool log_obs_terms_ = false;
    bool log_obs_ = true;
    bool log_action_raw_ = true;
    bool log_action_q_des_ = false;
    bool log_robot_state_ = true;
    bool log_commands_ = true;
    bool log_inference_outputs_ = true;
    bool log_torso_imu_ = true;
    bool log_lowcmd_ = true;
    bool logging_metadata_written_ = false;
    std::unique_ptr<G1LoggingTelemetry> logging_telemetry_;
    nlohmann::json logging_metadata_;
    std::chrono::steady_clock::time_point last_log_time;
    std::chrono::steady_clock::time_point start_time;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
    std::thread prewarm_thread_;
    std::atomic<bool> prewarm_started_{false};
    std::atomic<bool> prewarm_done_{false};
    std::atomic<bool> prewarmed_{false};
    std::uint64_t entry_inference_seq_ = 0;
    joint_filter::JointVectorFilterBank action_filter_;
    std::vector<float> filtered_action_;
};

REGISTER_FSM(State_RLBase)
