// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include "FSM/FSMState.h"
#include "utils/ExternalVelocity.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>
#include <array>
#include <set>
#include <vector>

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start() 
    {
        // Start From State_Passive
        currentState = states[0];
        currentState->enter();

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    ~CtrlFSM()
    {
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    void run_()
    {
        currentState->pre_run();
        apply_external_joystick_();
        FSMState::update_effective_joystick();
        update_external_velocity_();
        currentState->run();
        currentState->post_run();

        // Check if need to change state
        int nextStateMode = 0;
        std::string requested_state;
        const bool external_request =
            robot_automation::ExternalVelocity::instance().take_requested_state(requested_state);
        if (external_request)
        {
            const auto requested_id = FSMStringMap.right.find(requested_state);
            if (requested_id == FSMStringMap.right.end())
            {
                robot_automation::ExternalVelocity::instance().report_state_result(
                    false, "Unknown FSM state: " + requested_state);
            }
            else
            {
                nextStateMode = requested_id->second;
            }
        }
        else
        {
            for(int i(0); i<currentState->registered_checks.size(); i++)
            {
                if(currentState->registered_checks[i].first())
                {
                    nextStateMode = currentState->registered_checks[i].second;
                    break;
                }
            }
        }

        if(nextStateMode != 0 && !currentState->isState(nextStateMode))
        {
            for(auto & state : states)
            {
                if(state->isState(nextStateMode))
                {
                    spdlog::info("FSM: Change state from {} to {}", currentState->getStateString(), state->getStateString());
                    robot_automation::ExternalVelocity::instance().update("Transition", false, false);
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    update_external_velocity_();
                    robot_automation::ExternalVelocity::instance().report_state_result(
                        true, "Transitioned to " + state->getStateString());
                    break;
                }
            }
        }
        else if (external_request && nextStateMode != 0)
        {
            robot_automation::ExternalVelocity::instance().report_state_result(
                true, "Already in " + requested_state);
        }
    }

    void apply_external_joystick_()
    {
        std::vector<std::string> buttons;
        bool has_axes = false;
        std::array<float, 4> axes{0.0f, 0.0f, 0.0f, 0.0f};
        if (robot_automation::ExternalVelocity::instance().virtual_joystick(buttons, has_axes, axes))
        {
            FSMState::set_virtual_input(std::set<std::string>(buttons.begin(), buttons.end()),
                                        has_axes, axes[0], axes[1], axes[2], axes[3]);
        }
        else
        {
            FSMState::clear_virtual_input();
        }
    }

    void update_external_velocity_()
    {
        auto& joy = FSMState::lowstate->joystick;
        const bool manual = joy.LT.pressed || std::abs(joy.lx()) > 0.12f ||
            std::abs(joy.ly()) > 0.12f || std::abs(joy.rx()) > 0.12f;
        robot_automation::ExternalVelocity::instance().update(
            currentState->getStateString(), manual, !FSMState::lowstate->isTimeout());
    }

    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};