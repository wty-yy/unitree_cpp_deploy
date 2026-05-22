#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        // TODO: smooth and limit the velocity commands
        cmd = key_commands[key];
    }
    return cmd;
}

}

namespace
{
std::vector<joint_filter::JointFilterConfig> default_velocity_q_target_filters(std::size_t dof)
{
    const auto waist_joint_indices = joint_filter::default_waist_joint_indices(dof);
    if (waist_joint_indices.empty())
    {
        return {};
    }

    joint_filter::JointFilterConfig filter_cfg;
    filter_cfg.enabled = true;
    filter_cfg.type = joint_filter::JointFilterType::Lpf;
    filter_cfg.has_joint_indices = true;
    filter_cfg.joint_indices = waist_joint_indices;
    filter_cfg.lpf.alpha = 0.7f;
    return {filter_cfg};
}
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    const std::size_t dof = env->robot->data.joint_ids_map.size();
    const std::string filter_name = "State_RLBase(" + getStateString() + ") joint_filters.q_target";
    if (cfg["joint_filters"] && cfg["joint_filters"]["q_target"])
    {
        action_filter_.configure(
            cfg["joint_filters"]["q_target"],
            dof,
            filter_name,
            joint_filter::default_waist_joint_indices(dof));
    }
    else if (cfg["joint_filter"])
    {
        action_filter_.configure(
            cfg["joint_filter"],
            dof,
            filter_name,
            joint_filter::default_waist_joint_indices(dof));
    }
    else
    {
        action_filter_.configure(
            default_velocity_q_target_filters(dof),
            dof,
            filter_name,
            joint_filter::default_waist_joint_indices(dof));
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    action_filter_.apply(action);
    filtered_action_ = action;
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
