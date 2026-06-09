#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "FSM/state_preflight_utils.h"
#include <ctime>
#include <iomanip>
#include <sstream>
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

bool logging_option_enabled(const YAML::Node& cfg, const std::string& key, bool default_value = true)
{
    if (cfg["logging_options"] && cfg["logging_options"][key])
    {
        return cfg["logging_options"][key].as<bool>();
    }
    return default_value;
}

std::string observation_group_log_key(const std::string& group_name)
{
    if (group_name == "obs")
    {
        return "obs";
    }
    return "obs_" + group_name;
}

std::string observation_term_log_key(const isaaclab::ObservationTermSnapshot& snapshot)
{
    if (snapshot.group_name == "obs")
    {
        return "obs_" + snapshot.term_name;
    }
    return "obs_" + snapshot.group_name + "_" + snapshot.term_name;
}
}

FsmPreflightResult State_RLBase::preflight(const YAML::Node& cfg, const std::string& state_name)
{
    std::filesystem::path policy_dir;
    auto result = fsm_preflight::require_policy_dir(cfg, state_name, policy_dir);
    if (!result.enabled)
    {
        return result;
    }

    result = fsm_preflight::require_file(policy_dir / "params" / "deploy.yaml", "deploy yaml for " + state_name);
    if (!result.enabled)
    {
        return result;
    }
    return fsm_preflight::require_file(policy_dir / "exported" / "policy.onnx", "policy model for " + state_name);
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

    enable_logging = cfg["logging"] ? cfg["logging"].as<bool>() : true;
    log_obs_terms_ = logging_option_enabled(cfg, "obs_terms");
    log_obs_ = logging_option_enabled(cfg, "obs");
    log_action_raw_ = logging_option_enabled(cfg, "action_raw");
    log_action_q_des_ = logging_option_enabled(cfg, "action_q_des");
    log_robot_state_ = logging_option_enabled(cfg, "robot_state");
    log_commands_ = logging_option_enabled(cfg, "commands");
    log_inference_outputs_ = logging_option_enabled(cfg, "inference_outputs");

    if (enable_logging)
    {
        logging_dt = std::chrono::duration<double>(env->step_dt);
        if (cfg["logging_dt"] && !cfg["logging_dt"].IsNull())
        {
            logging_dt = std::chrono::duration<double>(cfg["logging_dt"].as<double>());
        }
        if (logging_dt.count() <= 0.0)
        {
            logging_dt = std::chrono::duration<double>(env->step_dt);
        }

        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        const std::string filename = "run_data_" + ss.str() + ".csv";
        const auto logs_dir = policy_dir / "logs";
        if (!std::filesystem::exists(logs_dir))
        {
            std::filesystem::create_directories(logs_dir);
        }

        const auto file_path = (logs_dir / filename).string();
        logger = std::make_unique<DataLogger>(file_path);
        spdlog::info("State_{} logging enabled. Saving to {} at {:.4f}s", state_string, file_path, logging_dt.count());

        start_time = std::chrono::steady_clock::now();
        last_log_time = start_time - std::chrono::duration_cast<std::chrono::steady_clock::duration>(logging_dt);
    }

    if (cfg["fixed_command"] && cfg["fixed_command"]["enabled"])
    {
        env->fixed_command_enabled = cfg["fixed_command"]["enabled"].as<bool>();
        if (env->fixed_command_enabled)
        {
            env->fixed_lin_vel_x = cfg["fixed_command"]["lin_vel_x"].as<float>();
            env->fixed_lin_vel_y = cfg["fixed_command"]["lin_vel_y"].as<float>();
            env->fixed_ang_vel_z = cfg["fixed_command"]["ang_vel_z"].as<float>();
            if (cfg["fixed_command"]["duration"])
            {
                env->fixed_command_duration = cfg["fixed_command"]["duration"].as<float>();
            }
            spdlog::info(
                "Fixed command enabled: lin_vel_x={:.2f}, lin_vel_y={:.2f}, ang_vel_z={:.2f}, duration={:.1f}s",
                env->fixed_lin_vel_x,
                env->fixed_lin_vel_y,
                env->fixed_ang_vel_z,
                env->fixed_command_duration);
            spdlog::info("Press [L2 + Y] to toggle fixed command execution");
        }
    }
}

void State_RLBase::run()
{
    if (!env)
    {
        return;
    }

    if (env->fixed_command_enabled)
    {
        auto& joy = lowstate->joystick;
        if (joy.LT.pressed && joy.Y.on_pressed)
        {
            env->fixed_command_active = !env->fixed_command_active;
            if (env->fixed_command_active)
            {
                env->fixed_command_start_time = std::chrono::steady_clock::now();
                if (env->fixed_command_duration > 0.0f)
                {
                    spdlog::info(
                        "Fixed command ACTIVATED for {:.1f}s: lin_vel_x={:.2f}, lin_vel_y={:.2f}, ang_vel_z={:.2f}",
                        env->fixed_command_duration,
                        env->fixed_lin_vel_x,
                        env->fixed_lin_vel_y,
                        env->fixed_ang_vel_z);
                }
                else
                {
                    spdlog::info(
                        "Fixed command ACTIVATED (indefinite): lin_vel_x={:.2f}, lin_vel_y={:.2f}, ang_vel_z={:.2f}",
                        env->fixed_lin_vel_x,
                        env->fixed_lin_vel_y,
                        env->fixed_ang_vel_z);
                }
            }
            else
            {
                spdlog::info("Fixed command DEACTIVATED, returning to joystick control");
            }
        }

        if (env->fixed_command_active && env->fixed_command_duration > 0.0f)
        {
            const auto elapsed = std::chrono::steady_clock::now() - env->fixed_command_start_time;
            const float elapsed_sec = std::chrono::duration<float>(elapsed).count();
            if (elapsed_sec >= env->fixed_command_duration)
            {
                env->fixed_command_active = false;
                spdlog::info(
                    "Fixed command COMPLETED after {:.1f}s, returning to joystick control",
                    elapsed_sec);
            }
        }
    }

    auto action = env->action_manager->processed_actions();
    action_filter_.apply(action);
    filtered_action_ = action;
    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    if (!enable_logging || !logger)
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_log_time < logging_dt)
    {
        return;
    }

    const auto snapshot = env->get_policy_logging_snapshot();
    const bool snapshot_required = log_obs_terms_ || log_obs_ || log_action_raw_ || log_inference_outputs_;
    const bool snapshot_ready =
        !snapshot.observation_groups.empty() ||
        !snapshot.observation_terms.empty() ||
        !snapshot.raw_action.empty() ||
        !snapshot.inference_results.empty();
    if (snapshot_required && !snapshot_ready)
    {
        return;
    }

    last_log_time = now;

    const std::chrono::duration<double> time_since_start = now - start_time;
    logger->add("time", time_since_start.count());

    const auto system_now = std::chrono::system_clock::now();
    const auto epoch_duration = system_now.time_since_epoch();
    const double unix_time =
        std::chrono::duration_cast<std::chrono::duration<double>>(epoch_duration).count();
    std::stringstream ss_unix;
    ss_unix << std::fixed << std::setprecision(2) << unix_time;
    logger->add("unix_time", ss_unix.str());

    const std::time_t now_c = std::chrono::system_clock::to_time_t(system_now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_duration).count() % 1000;
    std::stringstream ss_wall;
    ss_wall << std::put_time(std::localtime(&now_c), "%H:%M:%S")
            << '.'
            << std::setw(2)
            << std::setfill('0')
            << (ms / 10);
    logger->add("wall_time", ss_wall.str());

    if (log_obs_terms_)
    {
        for (const auto& term_snapshot : snapshot.observation_terms)
        {
            logger->add(observation_term_log_key(term_snapshot), term_snapshot.values);
        }
    }

    if (log_obs_)
    {
        for (const auto& group_snapshot : snapshot.observation_groups)
        {
            logger->add(observation_group_log_key(group_snapshot.first), group_snapshot.second);
        }
    }

    if (log_action_raw_ && !snapshot.raw_action.empty())
    {
        logger->add("action_raw", snapshot.raw_action);
    }

    if (log_action_q_des_)
    {
        logger->add("q_des", filtered_action_);
    }

    if (log_robot_state_)
    {
        const auto& joint_ids = env->robot->data.joint_ids_map;
        std::vector<float> q;
        std::vector<float> dq;
        std::vector<float> tau;
        std::vector<float> temp;
        q.reserve(joint_ids.size());
        dq.reserve(joint_ids.size());
        tau.reserve(joint_ids.size());
        temp.reserve(joint_ids.size() * 2);
        for (int i = 0; i < joint_ids.size(); ++i)
        {
            const int joint_id = static_cast<int>(joint_ids[i]);
            const auto& motor = lowstate->msg_.motor_state()[joint_id];
            q.push_back(motor.q());
            dq.push_back(motor.dq());
            tau.push_back(motor.tau_est());

            const auto& motor_temp = motor.temperature();
            temp.push_back(static_cast<float>(motor_temp[0]));
            temp.push_back(static_cast<float>(motor_temp[1]));
        }
        logger->add("q", q);
        logger->add("dq", dq);
        logger->add("tau", tau);
        logger->add("temp", temp);

        std::vector<float> imu_rpy(3);
        std::vector<float> imu_acc(3);
        std::vector<float> ang_vel(3);
        for (int i = 0; i < 3; ++i)
        {
            imu_rpy[i] = lowstate->msg_.imu_state().rpy()[i];
            imu_acc[i] = lowstate->msg_.imu_state().accelerometer()[i];
            ang_vel[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        logger->add("imu_rpy", imu_rpy);
        logger->add("imu_acc", imu_acc);
        logger->add("ang_vel", ang_vel);

        std::vector<float> imu_quat(4);
        for (int i = 0; i < 4; ++i)
        {
            imu_quat[i] = lowstate->msg_.imu_state().quaternion()[i];
        }
        logger->add("imu_quat", imu_quat);
    }

    if (log_inference_outputs_)
    {
        const auto weights_it = snapshot.inference_results.find("weights");
        if (weights_it != snapshot.inference_results.end())
        {
            logger->add("weight", weights_it->second);
        }

        const auto latent_it = snapshot.inference_results.find("latent");
        if (latent_it != snapshot.inference_results.end())
        {
            logger->add("latent", latent_it->second);
        }
    }

    if (log_commands_)
    {
        logger->add("cmd_ns_0", lowstate->joystick.ly());
        logger->add("cmd_ns_1", -lowstate->joystick.lx());
        logger->add("cmd_ns_2", -lowstate->joystick.rx());

        float fixed_0 = 0.0f;
        float fixed_1 = 0.0f;
        float fixed_2 = 0.0f;
        if (env->fixed_command_enabled && env->fixed_command_active)
        {
            fixed_0 = env->fixed_lin_vel_x;
            fixed_1 = env->fixed_lin_vel_y;
            fixed_2 = env->fixed_ang_vel_z;
        }
        logger->add("cmd_fixed_0", fixed_0);
        logger->add("cmd_fixed_1", fixed_1);
        logger->add("cmd_fixed_2", fixed_2);
    }

    logger->write();
}
