#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    spdlog::info("Initializing State_{}...", state_string);
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 2.0); },
            (int)FSMMode::Passive
        )
    );

    // Initialize logger
    if (cfg["logging"] && cfg["logging"].as<bool>()) {
        enable_logging = true;
        if (cfg["logging_dt"]) {
            logging_dt = std::chrono::duration<double>(cfg["logging_dt"].as<double>());
        }
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        std::string filename = "run_data_" + ss.str() + ".csv";
        auto logs_dir = policy_dir / "logs";
        if (!std::filesystem::exists(logs_dir)) {
            std::filesystem::create_directories(logs_dir);
        }
        auto file_path = (logs_dir / filename).string();
        logger = std::make_unique<DataLogger>(file_path);
        spdlog::info("Logging enabled. Saving to {}", file_path);
        
        start_time = std::chrono::steady_clock::now();
        last_log_time = start_time - std::chrono::duration_cast<std::chrono::steady_clock::duration>(logging_dt);
    }

    // Initialize fixed command settings
    if (cfg["fixed_command"] && cfg["fixed_command"]["enabled"]) {
        env->fixed_command_enabled = cfg["fixed_command"]["enabled"].as<bool>();
        if (env->fixed_command_enabled) {
            env->fixed_lin_vel_x = cfg["fixed_command"]["lin_vel_x"].as<float>();
            env->fixed_lin_vel_y = cfg["fixed_command"]["lin_vel_y"].as<float>();
            env->fixed_ang_vel_z = cfg["fixed_command"]["ang_vel_z"].as<float>();
            if (cfg["fixed_command"]["duration"]) {
                env->fixed_command_duration = cfg["fixed_command"]["duration"].as<float>();
            }
            spdlog::info("Fixed command enabled: lin_vel_x={:.2f}, lin_vel_y={:.2f}, ang_vel_z={:.2f}, duration={:.1f}s",
                env->fixed_lin_vel_x, env->fixed_lin_vel_y, env->fixed_ang_vel_z, env->fixed_command_duration);
            spdlog::info("Press [L2 + Y] to toggle fixed command execution");
        }
    }
}

void State_RLBase::run()
{
    // Check for L2 + Y to toggle fixed command execution
    if (env->fixed_command_enabled) {
        auto & joy = lowstate->joystick;
        if (joy.LT.pressed && joy.Y.on_pressed) {
            env->fixed_command_active = !env->fixed_command_active;
            if (env->fixed_command_active) {
                env->fixed_command_start_time = std::chrono::steady_clock::now();
                if (env->fixed_command_duration > 0) {
                    spdlog::info("Fixed command ACTIVATED for {:.1f}s: lin_vel_x={:.2f}, lin_vel_y={:.2f}, ang_vel_z={:.2f}",
                        env->fixed_command_duration, env->fixed_lin_vel_x, env->fixed_lin_vel_y, env->fixed_ang_vel_z);
                } else {
                    spdlog::info("Fixed command ACTIVATED (indefinite): lin_vel_x={:.2f}, lin_vel_y={:.2f}, ang_vel_z={:.2f}",
                        env->fixed_lin_vel_x, env->fixed_lin_vel_y, env->fixed_ang_vel_z);
                }
            } else {
                spdlog::info("Fixed command DEACTIVATED, returning to joystick control");
            }
        }

        // Check duration timeout
        if (env->fixed_command_active && env->fixed_command_duration > 0) {
            auto elapsed = std::chrono::steady_clock::now() - env->fixed_command_start_time;
            float elapsed_sec = std::chrono::duration<float>(elapsed).count();
            if (elapsed_sec >= env->fixed_command_duration) {
                env->fixed_command_active = false;
                spdlog::info("Fixed command COMPLETED after {:.1f}s, returning to joystick control", elapsed_sec);
            }
        }
    }

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    // Logging
    if (enable_logging && logger) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_log_time >= logging_dt) {
            last_log_time = now;
            
            std::chrono::duration<double> time_since_start = now - start_time;
            logger->add("time", time_since_start.count());
            
            auto system_now = std::chrono::system_clock::now();
            auto duration = system_now.time_since_epoch();
            double unix_time = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
            std::stringstream ss_unix;
            ss_unix << std::fixed << std::setprecision(2) << unix_time;
            logger->add("unix_time", ss_unix.str());

            std::time_t now_c = std::chrono::system_clock::to_time_t(system_now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
            std::stringstream ss_wall;
            ss_wall << std::put_time(std::localtime(&now_c), "%H:%M:%S") << '.' << std::setw(2) << std::setfill('0') << (ms / 10);
            logger->add("wall_time", ss_wall.str());

            logger->add("q_des", action);
            std::vector<float> q, dq, tau, temp;
            for (int i(0); i < 12; ++i) {
                q.push_back(lowstate->msg_.motor_state()[i].q());
                dq.push_back(lowstate->msg_.motor_state()[i].dq());
                tau.push_back(lowstate->msg_.motor_state()[i].tau_est());
                temp.push_back(lowstate->msg_.motor_state()[i].temperature());
            }
            logger->add("q", q);
            logger->add("dq", dq);
            logger->add("tau", tau);
            logger->add("temp", temp);

            std::vector<float> rpy(3), acc(3), gyro(3);
            for (int i(0); i < 3; ++i) {
                rpy[i] = lowstate->msg_.imu_state().rpy()[i];
                acc[i] = lowstate->msg_.imu_state().accelerometer()[i];
                gyro[i] = lowstate->msg_.imu_state().gyroscope()[i];
            }
            logger->add("imu_rpy", rpy);
            logger->add("imu_acc", acc);
            logger->add("ang_vel", gyro);

            std::vector<float> foot_force(4);
            for (int i(0); i < 4; ++i) {
                foot_force[i] = lowstate->msg_.foot_force()[i];
            }
            logger->add("foot_force", foot_force);

            std::vector<float> foot_contacts(4);
            for (int i(0); i < 4; ++i) {
                foot_contacts[i] = (foot_force[i] > 10.0f) ? 1.0f : 0.0f;
            }
            logger->add("foot_contact", foot_contacts);

            if (env->last_inference_results.count("weights")) {
                logger->add("weight", env->last_inference_results["weights"]);
            }
            if (env->last_inference_results.count("latent")) {
                logger->add("latent", env->last_inference_results["latent"]);
            }

            // Joystick commands (no scaling)
            logger->add("cmd_ns_0", lowstate->joystick.ly());
            logger->add("cmd_ns_1", -lowstate->joystick.lx());
            logger->add("cmd_ns_2", -lowstate->joystick.rx());

            std::vector<float> odom_pos(3), odom_vel(3);
            auto position_data = sportmodestate->msg_.position();
            auto velocity_data = sportmodestate->msg_.velocity();
            for (int i(0); i < 3; ++i) {
                odom_pos[i] = position_data[i];
                odom_vel[i] = velocity_data[i];
            }
            logger->add("odom_pos", odom_pos);
            logger->add("odom_vel", odom_vel);

            logger->write();
        }
    }
}