// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include "FSM/FSMOverlayState.h"
#include "FSM/FSMState.h"
#include "unitree_joystick_dsl.hpp"
#include <functional>
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>
#include <vector>
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

            const auto state_cfg = cfg[fsm_name];
            const auto preflight = fsm_class->second.preflight
                ? fsm_class->second.preflight(state_cfg, fsm_name)
                : FsmPreflightResult{};
            if (!preflight.enabled)
            {
                disabled_states_[id] = preflight.reason;
                spdlog::warn("FSM: State {} disabled: {}", fsm_name, preflight.reason);
                continue;
            }

            auto state_instance = fsm_class->second.factory(id, fsm_name);
            add(state_instance);
        }

        register_overlays_();

        if (states.empty())
        {
            throw std::runtime_error("FSM: no enabled states available");
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
            if (is_disabled_state_(nextStateMode))
            {
                warn_disabled_state_(nextStateMode);
                return;
            }

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

    bool is_disabled_state_(int state_id) const
    {
        return disabled_states_.count(state_id) > 0;
    }

    void warn_disabled_state_(int state_id)
    {
        std::string state_name = std::to_string(state_id);
        if (FSMStringMap.left.count(state_id))
        {
            state_name = FSMStringMap.left.at(state_id);
        }
        spdlog::warn(
            "FSM: transition to disabled state {} ignored: {}",
            state_name,
            disabled_states_.at(state_id));
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

        for (const auto& disabled_overlay : disabled_overlay_triggers_)
        {
            if (disabled_overlay.trigger && disabled_overlay.trigger(FSMState::lowstate->joystick))
            {
                warn_disabled_state_(disabled_overlay.target_state_id);
            }
        }
    }

    void register_overlays_()
    {
        if (!overlay_cfg_)
        {
            return;
        }

        for (auto it = overlay_cfg_.begin(); it != overlay_cfg_.end(); ++it)
        {
            const std::string overlay_name = it->first.as<std::string>();
            const auto overlay = it->second;
            const std::string overlay_type = overlay_type_name_(overlay_name, overlay);
            auto overlay_class = getFsmOverlayMap().find("OverlayState_" + overlay_type);
            if (overlay_class == getFsmOverlayMap().end())
            {
                throw std::runtime_error("FSM: Unknown overlay type " + overlay_type);
            }

            const std::string target_state = overlay["target_state"]
                ? overlay["target_state"].as<std::string>()
                : "";
            if (target_state.empty())
            {
                spdlog::warn("FSM: Overlay {} disabled: target_state is missing", overlay_name);
                continue;
            }

            const auto target_id_it = FSMStringMap.right.find(target_state);
            if (target_id_it == FSMStringMap.right.end())
            {
                spdlog::warn("FSM: Overlay {} disabled: target state {} is not registered", overlay_name, target_state);
                continue;
            }

            if (disabled_states_.count(target_id_it->second))
            {
                spdlog::warn(
                    "FSM: Overlay {} disabled: target state {} is disabled: {}",
                    overlay_name,
                    target_state,
                    disabled_states_.at(target_id_it->second));
                register_disabled_overlay_trigger_(overlay, target_id_it->second);
                continue;
            }

            overlay_states_.push_back(overlay_class->second(overlay));
            spdlog::info("FSM: Registered overlay {} as {}", overlay_name, overlay_type);
        }
    }

    std::string overlay_type_name_(const std::string& overlay_name, const YAML::Node& overlay) const
    {
        if (overlay["type"])
        {
            return overlay["type"].as<std::string>();
        }

        constexpr std::string_view prefix = "Overlay";
        if (overlay_name.rfind(prefix, 0) == 0)
        {
            const auto suffix = overlay_name.substr(prefix.size());
            if (suffix == "BeyondMimic")
            {
                return "Mimic";
            }
            return suffix;
        }
        return overlay_name;
    }

    void register_disabled_overlay_trigger_(const YAML::Node& overlay, int target_state_id)
    {
        std::string open_expr = "RT + A.on_pressed";
        if (overlay["open"])
        {
            open_expr = overlay["open"].as<std::string>();
        }
        if (overlay["gamepad_map"] && overlay["gamepad_map"]["open"])
        {
            open_expr = overlay["gamepad_map"]["open"].as<std::string>();
        }

        unitree::common::dsl::Parser parser(open_expr);
        auto ast = parser.Parse();
        disabled_overlay_triggers_.push_back(DisabledOverlayTrigger{
            target_state_id,
            unitree::common::dsl::Compile(*ast)
        });
    }

    struct DisabledOverlayTrigger
    {
        int target_state_id{0};
        std::function<bool(const unitree::common::UnitreeJoystick&)> trigger;
    };

    std::shared_ptr<BaseState> currentState;
    std::vector<std::shared_ptr<FSMOverlayState>> overlay_states_;
    std::shared_ptr<FSMOverlayState> active_overlay_;
    std::vector<DisabledOverlayTrigger> disabled_overlay_triggers_;
    std::unordered_map<int, std::string> disabled_states_;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};
