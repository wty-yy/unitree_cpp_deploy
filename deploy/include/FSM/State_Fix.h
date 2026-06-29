// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"

class State_Fix : public FSMState
{
public:
    State_Fix(int state, std::string state_string = "Fix") 
    : FSMState(state, state_string) 
    {
        auto cfg = param::config["FSM"][state_string];
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        ts_ = cfg["ts"].as<std::vector<float>>();
        qs_ = cfg["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());

        if(cfg["auto_transition"])
        {
            std::string target_fsm = cfg["auto_transition"].as<std::string>();
            if(FSMStringMap.right.count(target_fsm))
            {
                int fsm_id = FSMStringMap.right.at(target_fsm);
                registered_checks.emplace_back(
                    std::make_pair(
                        [this]()->bool {
                            double t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
                            return t >= ts_.back();
                        },
                        fsm_id
                    )
                );
            }
            else
            {
                spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
            }
        }
    }

    void enter()
    {
        // set gain
        for(int i(0); i < kp_.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp_[i];
            motor.kd() = kd_[i];
            motor.dq() = motor.tau() = 0;
        }


        // set initial position
        std::vector<float> q0;
        for(int i(0); i < kp_.size(); ++i) {
            q0.push_back(lowcmd->msg_.motor_cmd()[i].q());
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);
        
        for(int i(0); i < q.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    double t0_;
    std::vector<float> kp_;
    std::vector<float> kd_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
};

REGISTER_FSM(State_Fix)
