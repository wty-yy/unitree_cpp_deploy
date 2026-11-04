// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "DataLogger.h"
#include "utils/Go2LoggingTelemetry.h"
#include "utils/PolicySelectionPublisher.h"
#include <chrono>
#include <atomic>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        if (!env) {
            spdlog::warn("State_RLBase::enter: Environment is null. Skipping initialization.");
            return;
        }

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
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                if (policy_selection_) policy_selection_->publish(env->last_inference_results);

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
        if (policy_selection_) policy_selection_->clear();
    }

private:
    void start_logging();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<DataLogger> logger;
    std::unique_ptr<utils::PolicySelectionPublisher> policy_selection_;
    bool enable_logging = false;
    std::chrono::duration<double> logging_dt{0.02};
    bool log_obs_terms_ = false;
    bool log_obs_ = true;
    bool log_action_raw_ = true;
    bool log_action_q_des_ = false;
    bool log_robot_state_ = true;
    bool log_commands_ = true;
    bool log_inference_outputs_ = true;
    bool log_lowcmd_ = true;
    bool logging_metadata_written_ = false;
    std::unique_ptr<Go2LoggingTelemetry> logging_telemetry_;
    nlohmann::json logging_metadata_;
    std::chrono::steady_clock::time_point last_log_time;
    std::chrono::steady_clock::time_point start_time;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
};

REGISTER_FSM(State_RLBase)
