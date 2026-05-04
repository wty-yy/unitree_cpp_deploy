#pragma once

#include "FSM/FSMOverlayState.h"
#include "FSM/FSMState.h"
#include "utils/motion_selection.h"
#include "utils/path_file_manager.h"
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

class OverlayState_Mimic : public FSMOverlayState
{
public:
    explicit OverlayState_Mimic(const YAML::Node& cfg)
    {
        target_state_name_ = cfg["target_state"] ? cfg["target_state"].as<std::string>() : "BeyondMimic";
        load_motion_candidates(cfg);
        if (!motion_selector_ || motion_selector_->empty())
        {
            throw std::runtime_error("OverlayState_Mimic: no csv motion files found from 'motion_files'");
        }

        if (cfg["open"]) key_cfg_.open = cfg["open"].as<std::string>();
        if (cfg["select_next"]) key_cfg_.select_next = cfg["select_next"].as<std::string>();
        if (cfg["select_prev"]) key_cfg_.select_prev = cfg["select_prev"].as<std::string>();
        if (cfg["confirm"]) key_cfg_.confirm = cfg["confirm"].as<std::string>();
        if (cfg["cancel"]) key_cfg_.cancel = cfg["cancel"].as<std::string>();

        const auto gamepad_map = cfg["gamepad_map"];
        if (gamepad_map)
        {
            if (gamepad_map["open"]) key_cfg_.open = gamepad_map["open"].as<std::string>();
            if (gamepad_map["select_next"]) key_cfg_.select_next = gamepad_map["select_next"].as<std::string>();
            if (gamepad_map["select_prev"]) key_cfg_.select_prev = gamepad_map["select_prev"].as<std::string>();
            if (gamepad_map["confirm"]) key_cfg_.confirm = gamepad_map["confirm"].as<std::string>();
            if (gamepad_map["cancel"]) key_cfg_.cancel = gamepad_map["cancel"].as<std::string>();
        }

        auto compile = [](const std::string& expr) {
            unitree::common::dsl::Parser p(expr);
            auto ast = p.Parse();
            return unitree::common::dsl::Compile(*ast);
        };

        open_trigger_ = compile(key_cfg_.open);
        select_next_trigger_ = compile(key_cfg_.select_next);
        select_prev_trigger_ = compile(key_cfg_.select_prev);
        confirm_trigger_ = compile(key_cfg_.confirm);
        cancel_trigger_ = compile(key_cfg_.cancel);
    }

    std::string name() const override
    {
        return "OverlayState_Mimic";
    }

    bool should_activate() const override
    {
        if (!FSMState::lowstate || !open_trigger_)
        {
            return false;
        }
        return open_trigger_(FSMState::lowstate->joystick);
    }

    void on_activate(const std::string& host_state_name) override
    {
        host_state_name_ = host_state_name;
        motion_selector_->set_index(last_selected_motion_index_);

        spdlog::info("OverlayState_Mimic: activated on host state '{}'.", host_state_name_);
        spdlog::info("OverlayState_Mimic: found {} csv motion files under configured paths.", motion_selector_->size());
        for (std::size_t i = 0; i < motion_selector_->size(); ++i)
        {
            const auto& file = motion_selector_->files()[i];
            const std::string marker = (i == motion_selector_->index()) ? " <== default" : "";
            spdlog::info("  [{}] {}{}", i + 1, file.string(), marker);
        }
        spdlog::info("OverlayState_Mimic controls: UP=prev, DOWN=next, A=confirm and execute BeyondMimic, B=cancel and keep current state.");
        std::cout << motion_selector_->render_live_line() << std::flush;
    }

    void on_update() override
    {
        if (!FSMState::lowstate)
        {
            return;
        }

        const auto& joy = FSMState::lowstate->joystick;
        if (cancel_trigger_ && cancel_trigger_(joy))
        {
            std::cout << std::endl;
            spdlog::info("OverlayState_Mimic: cancelled, keep host state '{}'.", host_state_name_);
            finish(0);
            return;
        }

        if (select_next_trigger_ && select_next_trigger_(joy))
        {
            if (motion_selector_->move_next())
            {
                std::cout << motion_selector_->render_live_line() << std::flush;
            }
        }
        else if (select_prev_trigger_ && select_prev_trigger_(joy))
        {
            if (motion_selector_->move_prev())
            {
                std::cout << motion_selector_->render_live_line() << std::flush;
            }
        }
        else if (confirm_trigger_ && confirm_trigger_(joy))
        {
            std::cout << std::endl;
            last_selected_motion_index_ = motion_selector_->index();
            confirmed_motion_file_ = motion_selector_->current();
            has_confirmed_motion_ = true;

            if (!FSMStringMap.right.count(target_state_name_))
            {
                throw std::runtime_error("OverlayState_Mimic: unknown target_state " + target_state_name_);
            }
            const int target_state_id = FSMStringMap.right.at(target_state_name_);

            spdlog::info("OverlayState_Mimic confirmed [{}/{}]: '{}', request transition to {}.",
                         motion_selector_->index() + 1,
                         motion_selector_->size(),
                         motion_selector_->current().filename().string(),
                         target_state_name_);
            finish(target_state_id);
        }
    }

    void on_deactivate() override
    {
        // No-op
    }

    static bool has_confirmed_motion()
    {
        return has_confirmed_motion_;
    }

    static const std::filesystem::path& confirmed_motion_file()
    {
        return confirmed_motion_file_;
    }

private:
    struct KeyConfig
    {
        std::string open = "RT + A.on_pressed";
        std::string select_next = "down.on_pressed";
        std::string select_prev = "up.on_pressed";
        std::string confirm = "A.on_pressed";
        std::string cancel = "B.on_pressed";
    };

    void load_motion_candidates(const YAML::Node& cfg)
    {
        std::vector<std::filesystem::path> configured_paths;
        if (cfg["motion_files"])
        {
            if (cfg["motion_files"].IsScalar())
            {
                configured_paths.emplace_back(cfg["motion_files"].as<std::string>());
            }
            else if (cfg["motion_files"].IsSequence())
            {
                const auto paths = cfg["motion_files"].as<std::vector<std::string>>();
                configured_paths.reserve(paths.size());
                for (const auto& p : paths)
                {
                    configured_paths.emplace_back(p);
                }
            }
        }

        if (configured_paths.empty() && cfg["motion_file"])
        {
            configured_paths.emplace_back(cfg["motion_file"].as<std::string>());
            spdlog::warn("OverlayState_Mimic: 'motion_file' is deprecated; please switch to 'motion_files'.");
        }

        const auto files = PathFileManager::collect_csv_files(configured_paths, param::proj_dir);
        motion_selector_ = std::make_unique<MotionSelection>(files, last_selected_motion_index_);
    }

private:
    KeyConfig key_cfg_{};
    std::string target_state_name_{"BeyondMimic"};
    std::string host_state_name_{};
    std::unique_ptr<MotionSelection> motion_selector_;

    std::function<bool(const unitree::common::UnitreeJoystick&)> open_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> select_next_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> select_prev_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> confirm_trigger_;
    std::function<bool(const unitree::common::UnitreeJoystick&)> cancel_trigger_;

    inline static std::size_t last_selected_motion_index_{0};
    inline static bool has_confirmed_motion_{false};
    inline static std::filesystem::path confirmed_motion_file_{};
};
