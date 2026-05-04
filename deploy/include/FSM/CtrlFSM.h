// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include "FSM/OverlayState_Mimic.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

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
        overlay_cfg_ = cfg["overlay"];
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

        if (overlay_cfg_ && overlay_cfg_["OverlayBeyondMimic"])
        {
            overlay_states_.push_back(std::make_shared<OverlayState_Mimic>(overlay_cfg_["OverlayBeyondMimic"]));
            spdlog::info("FSM: Registered overlay OverlayBeyondMimic");
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
        if (active_overlay_)
        {
            active_overlay_->deactivate();
            active_overlay_.reset();
        }
        overlay_states_.clear();
        states.clear();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;
    YAML::Node overlay_cfg_;

    void run_()
    {
        currentState->pre_run();
        currentState->run();
        update_overlay_();
        currentState->post_run();
        
        // Check if need to change state
        int nextStateMode = 0;
        if (active_overlay_ && active_overlay_->finished() && active_overlay_->requested_state_id() != 0)
        {
            nextStateMode = active_overlay_->requested_state_id();
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
                    if (active_overlay_)
                    {
                        active_overlay_->deactivate();
                        active_overlay_.reset();
                    }
                    currentState->exit();
                    currentState = state;
                    currentState->enter();
                    break;
                }
            }
        }
    }

    void update_overlay_()
    {
        if (active_overlay_)
        {
            active_overlay_->update();
            if (active_overlay_->finished() && active_overlay_->requested_state_id() == 0)
            {
                active_overlay_->deactivate();
                active_overlay_.reset();
            }
            return;
        }

        for (const auto& overlay : overlay_states_)
        {
            if (overlay->should_activate())
            {
                active_overlay_ = overlay;
                active_overlay_->activate(currentState->getStateString());
                break;
            }
        }
    }

    std::shared_ptr<BaseState> currentState;
    std::vector<std::shared_ptr<FSMOverlayState>> overlay_states_;
    std::shared_ptr<FSMOverlayState> active_overlay_;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};
