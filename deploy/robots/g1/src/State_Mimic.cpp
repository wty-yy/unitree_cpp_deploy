#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/commands/motion_command.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;

namespace
{

Eigen::Quaternionf torso_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    using G1Type = unitree::BaseArticulation<LowState_t::SharedPtr>;
    G1Type* robot = dynamic_cast<G1Type*>(env->robot.get());

    auto root_quat = env->robot->data.root_quat_w;
    auto& motors = robot->lowstate->msg_.motor_state();

    Eigen::Quaternionf torso_quat = root_quat
        * Eigen::AngleAxisf(motors[12].q(), Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(motors[13].q(), Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(motors[14].q(), Eigen::Vector3f::UnitY());
    return torso_quat;
}

Eigen::Quaternionf anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    const auto root_quat = loader->root_quaternion();
    const auto joint_pos = loader->joint_pos();
    Eigen::Quaternionf torso_quat = root_quat
        * Eigen::AngleAxisf(joint_pos[12], Eigen::Vector3f::UnitZ())
        * Eigen::AngleAxisf(joint_pos[13], Eigen::Vector3f::UnitX())
        * Eigen::AngleAxisf(joint_pos[14], Eigen::Vector3f::UnitY());
    return torso_quat;
}

} // namespace

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_joint_pos)
{
    auto& robot = env->robot;
    auto& loader = State_Mimic::motion;
    auto& ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_pos();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for (int i = 0; i < data_dfs.size(); ++i)
    {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_joint_vel)
{
    auto& robot = env->robot;
    auto& loader = State_Mimic::motion;
    auto& ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_vel();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for (int i = 0; i < data_dfs.size(); ++i)
    {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_command)
{
    auto& robot = env->robot;
    auto& loader = State_Mimic::motion;
    auto& ids = robot->data.joint_ids_map;

    auto pos_dfs = loader->joint_pos();
    Eigen::VectorXf pos_bfs = Eigen::VectorXf::Zero(pos_dfs.size());
    for (int i = 0; i < pos_dfs.size(); ++i)
    {
        pos_bfs(i) = pos_dfs[ids[i]];
    }

    auto vel_dfs = loader->joint_vel();
    Eigen::VectorXf vel_bfs = Eigen::VectorXf::Zero(vel_dfs.size());
    for (int i = 0; i < vel_dfs.size(); ++i)
    {
        vel_bfs(i) = vel_dfs[ids[i]];
    }

    std::vector<float> data;
    data.insert(data.end(), pos_bfs.data(), pos_bfs.data() + pos_bfs.size());
    data.insert(data.end(), vel_bfs.data(), vel_bfs.data() + vel_bfs.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto real_quat_w = torso_quat_w(env);
    auto ref_quat_w = anchor_quat_w(State_Mimic::motion);

    auto rot_ = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    auto rot = rot_.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

} // namespace mdp
} // namespace isaaclab

State_Mimic::State_Mimic(int state_mode, std::string state_string)
    : FSMState(state_mode, std::move(state_string))
{
    auto cfg = param::config["FSM"][getStateString()];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    const auto deploy_rel = cfg["deploy_yaml"] ? cfg["deploy_yaml"].as<std::string>() : "params/deploy.yaml";
    const auto onnx_rel = cfg["onnx_model"] ? cfg["onnx_model"].as<std::string>() : "exported/policy.onnx";

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute())
    {
        motion_file = param::proj_dir / motion_file;
    }

    motion_ = std::make_shared<MotionLoader_>(motion_file.string(), cfg["fps"].as<float>());
    spdlog::info("State_Mimic: loaded motion '{}' with duration {:.2f}s",
                 motion_file.stem().string(), motion_->duration);
    motion = motion_;

    time_range_[0] = cfg["time_start"] && !cfg["time_start"].IsNull()
        ? std::clamp(cfg["time_start"].as<float>(), 0.0f, motion_->duration)
        : 0.0f;
    time_range_[1] = cfg["time_end"] && !cfg["time_end"].IsNull()
        ? std::clamp(cfg["time_end"].as<float>(), 0.0f, motion_->duration)
        : motion_->duration;
    reference_time_.store(time_range_[0]);

    env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile((policy_dir / deploy_rel).string()),
        articulation
    );
    env_->alg = std::make_unique<isaaclab::OrtRunner>((policy_dir / onnx_rel).string());

    const std::string finished_state = cfg["finished_transition"]
        ? cfg["finished_transition"].as<std::string>()
        : "Velocity_Y";
    if (!FSMStringMap.right.count(finished_state))
    {
        throw std::runtime_error("State_Mimic: unknown finished_transition target " + finished_state);
    }
    finished_state_id_ = FSMStringMap.right.at(finished_state);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool { return motion_finished_.load(); },
            finished_state_id_
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool { return isaaclab::mdp::bad_orientation(env_.get(), 1.0f); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Mimic::enter()
{
    if (!env_)
    {
        spdlog::warn("State_Mimic::enter: env is null.");
        return;
    }

    for (int i = 0; i < env_->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env_->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env_->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    motion = motion_;
    env_->robot->update();
    reset_motion_state();

    policy_thread_running_ = true;
    policy_thread_ = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(env_->step_dt));
        auto sleep_till = clock::now() + dt;

        while (policy_thread_running_)
        {
            env_->robot->update();

            float time_to_sample = time_range_[0];
            if (execute_motion_.load())
            {
                time_to_sample = reference_time_.load();
            }
            motion_->update(time_to_sample);
            env_->step();

            if (execute_motion_.load())
            {
                const float next_time = std::min(reference_time_.load() + static_cast<float>(env_->step_dt), time_range_[1]);
                reference_time_.store(next_time);
                if (next_time >= time_range_[1])
                {
                    motion_finished_.store(true);
                }
            }

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    });
}

void State_Mimic::run()
{
    if (!env_)
    {
        return;
    }

    auto action = env_->action_manager->processed_actions();
    for (int i = 0; i < env_->robot->data.joint_ids_map.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[env_->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

void State_Mimic::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable())
    {
        policy_thread_.join();
    }
}

void State_Mimic::reset_motion_state()
{
    execute_motion_ = true;
    reference_time_.store(time_range_[0]);
    motion_finished_.store(false);

    auto ref_yaw = isaaclab::yawQuaternion(motion_->root_quaternion()).toRotationMatrix();
    auto robot_yaw = isaaclab::yawQuaternion(torso_quat_w(env_.get())).toRotationMatrix();
    init_quat = robot_yaw * ref_yaw.transpose();
    motion_->reset(env_->robot->data, time_range_[0]);
    env_->reset();
}

State_Mimic::MotionLoader_::MotionLoader_(std::string motion_file, float fps)
    : dt(1.0f / fps)
{
    auto loader = isaaclab::MotionLoader(motion_file, fps);
    num_frames = loader.num_frames;
    duration = loader.duration;
    root_positions = std::move(loader.root_positions);
    root_quaternions = std::move(loader.root_quaternions);
    dof_positions = std::move(loader.dof_positions);
    dof_velocities = std::move(loader.dof_velocities);
    update(0.0f);
}

void State_Mimic::MotionLoader_::update(float time)
{
    float phase = std::clamp(time / duration, 0.0f, 1.0f);
    index_0_ = std::round(phase * (num_frames - 1));
    index_1_ = std::min(index_0_ + 1, num_frames - 1);
    blend_ = std::round((time - index_0_ * dt) / dt * 1e5f) / 1e5f;
}

void State_Mimic::MotionLoader_::reset(const isaaclab::ArticulationData& data, float t)
{
    update(t);
    auto init_to_anchor = isaaclab::yawQuaternion(this->root_quaternion()).toRotationMatrix();
    auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
    world_to_init_ = world_to_anchor * init_to_anchor.transpose();
}

Eigen::VectorXf State_Mimic::MotionLoader_::joint_pos()
{
    return dof_positions[index_0_] * (1 - blend_) + dof_positions[index_1_] * blend_;
}

Eigen::VectorXf State_Mimic::MotionLoader_::root_position()
{
    return root_positions[index_0_] * (1 - blend_) + root_positions[index_1_] * blend_;
}

Eigen::VectorXf State_Mimic::MotionLoader_::joint_vel()
{
    return dof_velocities[index_0_] * (1 - blend_) + dof_velocities[index_1_] * blend_;
}

Eigen::Quaternionf State_Mimic::MotionLoader_::root_quaternion()
{
    return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]);
}
