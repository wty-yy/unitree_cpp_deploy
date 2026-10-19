#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <array>
#include <mutex>
#include <set>
#include <vector>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::effective_joystick()); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;

    // Virtual handheld input injected by the external AI. Every FSM/overlay/policy
    // trigger reads effective_joystick(), which mirrors the real remote and overlays
    // the AI-held buttons/axes, so edges and hold timers are generated automatically.
    using Joystick = unitree::common::UnitreeJoystick;

    static Joystick& effective_joystick() { return effective_; }

    static void set_virtual_input(const std::set<std::string>& held, bool has_axes,
                                  float lx, float ly, float rx, float ry)
    {
        std::lock_guard<std::mutex> guard(virtual_mutex_);
        virtual_held_ = held;
        virtual_has_axes_ = has_axes;
        virtual_axes_ = {lx, ly, rx, ry};
    }

    static void clear_virtual_input()
    {
        std::lock_guard<std::mutex> guard(virtual_mutex_);
        virtual_held_.clear();
        virtual_has_axes_ = false;
        virtual_axes_ = {0.0f, 0.0f, 0.0f, 0.0f};
    }

    static void update_effective_joystick()
    {
        if (!lowstate)
        {
            return;
        }
        std::set<std::string> held;
        bool has_axes = false;
        std::array<float, 4> axes{0.0f, 0.0f, 0.0f, 0.0f};
        {
            std::lock_guard<std::mutex> guard(virtual_mutex_);
            held = virtual_held_;
            has_axes = virtual_has_axes_;
            axes = virtual_axes_;
        }
        auto& real = lowstate->joystick;
        auto held_key = [&held](const char* name) { return held.count(name) != 0; };
        auto feed_button = [](unitree::common::Button<int>& out,
                              unitree::common::Button<int>& in, bool extra) {
            out((in() != 0 || extra) ? 1 : 0);
        };
        auto feed_axis = [](unitree::common::Axis& out,
                            unitree::common::Axis& in, bool hold) {
            out(hold ? 1.0f : in());
        };
        feed_button(effective_.back, real.back, held_key("back"));
        feed_button(effective_.start, real.start, held_key("start"));
        feed_button(effective_.LS, real.LS, held_key("ls"));
        feed_button(effective_.RS, real.RS, held_key("rs"));
        feed_button(effective_.LB, real.LB, held_key("lb"));
        feed_button(effective_.RB, real.RB, held_key("rb"));
        feed_button(effective_.A, real.A, held_key("a"));
        feed_button(effective_.B, real.B, held_key("b"));
        feed_button(effective_.X, real.X, held_key("x"));
        feed_button(effective_.Y, real.Y, held_key("y"));
        feed_button(effective_.up, real.up, held_key("up"));
        feed_button(effective_.down, real.down, held_key("down"));
        feed_button(effective_.left, real.left, held_key("left"));
        feed_button(effective_.right, real.right, held_key("right"));
        feed_button(effective_.F1, real.F1, held_key("f1"));
        feed_button(effective_.F2, real.F2, held_key("f2"));
        feed_axis(effective_.LT, real.LT, held_key("lt"));
        feed_axis(effective_.RT, real.RT, held_key("rt"));
        effective_.lx(has_axes ? axes[0] : real.lx());
        effective_.ly(has_axes ? axes[1] : real.ly());
        effective_.rx(has_axes ? axes[2] : real.rx());
        effective_.ry(has_axes ? axes[3] : real.ry());
    }

private:
    inline static Joystick effective_{};
    inline static std::mutex virtual_mutex_{};
    inline static std::set<std::string> virtual_held_{};
    inline static bool virtual_has_axes_{false};
    inline static std::array<float, 4> virtual_axes_{0.0f, 0.0f, 0.0f, 0.0f};
};