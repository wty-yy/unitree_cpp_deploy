#include "State_BFM.h"

#include "cnpy.h"
#include "unitree_articulation.h"
#include "unitree_joystick_dsl.hpp"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/envs/mdp/observations/observations.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace
{
std::string infer_task_type_from_name(const std::string& state_name)
{
    if (state_name.find("tracking") != std::string::npos || state_name.find("Tracking") != std::string::npos)
    {
        return "tracking";
    }
    if (state_name.find("reward") != std::string::npos || state_name.find("Reward") != std::string::npos)
    {
        return "reward";
    }
    return "goal";
}

enum class OnnxExecProvider
{
    TensorRT,
    CUDA,
    CPU
};

OnnxExecProvider append_best_provider(Ort::SessionOptions& options, int device_id, bool enable_tensorrt, bool enable_cuda)
{
    const OrtApi& api = Ort::GetApi();

    char** providers = nullptr;
    int provider_count = 0;
    OrtStatus* status = api.GetAvailableProviders(&providers, &provider_count);
    if (status != nullptr)
    {
        spdlog::warn("State_BFM: GetAvailableProviders failed: {}", api.GetErrorMessage(status));
        api.ReleaseStatus(status);
        return OnnxExecProvider::CPU;
    }

    bool has_tensorrt = false;
    bool has_cuda = false;
    for (int i = 0; i < provider_count; ++i)
    {
        const std::string provider_name = providers[i];
        if (provider_name == "TensorrtExecutionProvider")
        {
            has_tensorrt = true;
        }
        else if (provider_name == "CUDAExecutionProvider")
        {
            has_cuda = true;
        }
    }

    OrtStatus* release_status = api.ReleaseAvailableProviders(providers, provider_count);
    if (release_status != nullptr)
    {
        api.ReleaseStatus(release_status);
    }

    if (enable_tensorrt && has_tensorrt)
    {
        OrtTensorRTProviderOptions trt_options{};
        trt_options.device_id = device_id;
        status = api.SessionOptionsAppendExecutionProvider_TensorRT(options, &trt_options);
        if (status == nullptr)
        {
            return OnnxExecProvider::TensorRT;
        }
        spdlog::warn("State_BFM: append TensorRT provider failed: {}", api.GetErrorMessage(status));
        api.ReleaseStatus(status);
    }

    if (enable_cuda && has_cuda)
    {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = device_id;
        status = api.SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options);
        if (status == nullptr)
        {
            return OnnxExecProvider::CUDA;
        }
        spdlog::warn("State_BFM: append CUDA provider failed: {}", api.GetErrorMessage(status));
        api.ReleaseStatus(status);
    }

    return OnnxExecProvider::CPU;
}

std::vector<float> clamp_vec(const std::vector<float>& x, const std::vector<float>& lo, const std::vector<float>& hi)
{
    std::vector<float> y = x;
    for (std::size_t i = 0; i < y.size(); ++i)
    {
        y[i] = std::clamp(y[i], lo[i], hi[i]);
    }
    return y;
}

float vec_norm(const std::vector<float>& v)
{
    float sum2 = 0.0f;
    for (float x : v)
    {
        sum2 += x * x;
    }
    return std::sqrt(sum2);
}

} // namespace

State_BFM::State_BFM(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    if (!cfg)
    {
        throw std::runtime_error("State_BFM: missing config for state " + state_string);
    }

    load_policy_and_env(cfg);
    initialize_limits(cfg);
    load_task_context(cfg);
    load_key_config(cfg);

}

void State_BFM::load_policy_and_env(const YAML::Node& cfg)
{
    if (!cfg["policy_dir"] || cfg["policy_dir"].IsNull())
    {
        throw std::runtime_error("State_BFM: policy_dir is required");
    }

    const auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    const auto deploy_rel = cfg["deploy_yaml"] ? cfg["deploy_yaml"].as<std::string>() : "param/deploy.yaml";
    const auto onnx_rel = cfg["onnx_model"] ? cfg["onnx_model"].as<std::string>() : "exported/FBcprAuxModel.onnx";

    const auto deploy_path = policy_dir / deploy_rel;
    const auto onnx_path = policy_dir / onnx_rel;

    auto deploy_cfg = YAML::LoadFile(deploy_path.string());

    env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );

    action_scale_ = deploy_cfg["actions"]["JointPositionAction"]["scale"].as<std::vector<float>>();
    default_joint_pos_ = deploy_cfg["default_joint_pos"].as<std::vector<float>>();
    action_rescale_ = deploy_cfg["action_rescale"] ? deploy_cfg["action_rescale"].as<float>() : 5.0f;

    const bool prefer_cuda = cfg["onnx_cuda"] ? cfg["onnx_cuda"].as<bool>() : true;
    const bool prefer_tensorrt = cfg["onnx_tensorrt"] ? cfg["onnx_tensorrt"].as<bool>() : true;
    const int cuda_device_id = cfg["onnx_cuda_device"] ? cfg["onnx_cuda_device"].as<int>() : 0;

    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    const OnnxExecProvider selected_provider =
        append_best_provider(session_options_, cuda_device_id, prefer_tensorrt, prefer_cuda);

    session_ = std::make_unique<Ort::Session>(ort_env_, onnx_path.string().c_str(), session_options_);

    auto input_name = session_->GetInputNameAllocated(0, allocator_);
    onnx_input_name_ = input_name.get();

    output_names_str_.clear();
    output_names_.clear();
    const std::size_t num_outputs = session_->GetOutputCount();
    output_names_str_.reserve(num_outputs);
    output_names_.reserve(num_outputs);

    for (std::size_t i = 0; i < num_outputs; ++i)
    {
        auto out_name = session_->GetOutputNameAllocated(i, allocator_);
        output_names_str_.push_back(out_name.get());
    }
    for (auto& name : output_names_str_)
    {
        output_names_.push_back(name.c_str());
    }

    if (!output_names_str_.empty())
    {
        const auto it = std::find(output_names_str_.begin(), output_names_str_.end(), action_output_name_);
        if (it == output_names_str_.end())
        {
            action_output_name_ = output_names_str_[0];
        }
    }

    spdlog::info("State_BFM: loaded ONNX model: {}", onnx_path.string());
    if (selected_provider == OnnxExecProvider::TensorRT)
    {
        spdlog::info("State_BFM: execution provider: TensorRT");
    }
    else if (selected_provider == OnnxExecProvider::CUDA)
    {
        spdlog::info("State_BFM: execution provider: CUDA");
    }
    else
    {
        spdlog::warn("State_BFM: TensorRT/CUDA provider not available, fallback to CPU");
    }
}

void State_BFM::initialize_limits(const YAML::Node& cfg)
{
    const std::size_t dof = env_->robot->data.joint_ids_map.size();
    if (default_joint_pos_.size() != dof || action_scale_.size() != dof)
    {
        throw std::runtime_error("State_BFM: deploy.yaml action/default dimension mismatch");
    }

    joint_pos_lower_limit_.assign(dof, -std::numeric_limits<float>::max());
    joint_pos_upper_limit_.assign(dof, std::numeric_limits<float>::max());

    if (cfg["joint_pos_lower_limit"]) {
        joint_pos_lower_limit_ = cfg["joint_pos_lower_limit"].as<std::vector<float>>();
    }
    if (cfg["joint_pos_upper_limit"]) {
        joint_pos_upper_limit_ = cfg["joint_pos_upper_limit"].as<std::vector<float>>();
    }

    if (joint_pos_lower_limit_.size() != dof || joint_pos_upper_limit_.size() != dof)
    {
        throw std::runtime_error("State_BFM: joint position limit size mismatch");
    }

    last_action_.assign(dof, 0.0f);
    latest_q_target_ = default_joint_pos_;
}

void State_BFM::load_task_context(const YAML::Node& cfg)
{
    std::string task = cfg["task_type"] ? cfg["task_type"].as<std::string>() : infer_task_type_from_name(getStateString());
    if (task == "tracking")
    {
        task_type_ = TaskType::Tracking;
    }
    else if (task == "reward")
    {
        task_type_ = TaskType::Reward;
    }
    else
    {
        task_type_ = TaskType::Goal;
        task = "goal";
    }

    std::filesystem::path policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    std::string latent_rel;
    if (cfg["latent_file"])
    {
        latent_rel = cfg["latent_file"].as<std::string>();
    }
    else if (task_type_ == TaskType::Tracking)
    {
        latent_rel = "tracking_inference/zs_walking.npz";
    }
    else if (task_type_ == TaskType::Reward)
    {
        latent_rel = "reward_inference/reward_locomotion.npz";
    }
    else
    {
        latent_rel = "goal_inference/goal_reaching.npz";
    }

    std::filesystem::path latent_path = policy_dir / latent_rel;
    auto npz = cnpy::npz_load(latent_path.string());

    selected_latents_.clear();
    selected_latent_names_.clear();

    if (task_type_ == TaskType::Tracking)
    {
        const std::string tracking_key = cfg["tracking"]["key"] ? cfg["tracking"]["key"].as<std::string>() : "data";
        if (!npz.count(tracking_key))
        {
            throw std::runtime_error("State_BFM(tracking): key not found in npz: " + tracking_key);
        }

        auto arr = to_float_vector(npz[tracking_key]);
        const auto shape = npz[tracking_key].shape;
        if (shape.size() != 2 || shape[1] == 0)
        {
            throw std::runtime_error("State_BFM(tracking): expected 2D array [T, latent_dim]");
        }

        latent_dim_ = shape[1];
        tracking_ctx_.assign(shape[0], std::vector<float>(latent_dim_, 0.0f));
        for (std::size_t i = 0; i < shape[0]; ++i)
        {
            std::copy(arr.begin() + i * latent_dim_, arr.begin() + (i + 1) * latent_dim_, tracking_ctx_[i].begin());
        }

        t_start_ = cfg["tracking"]["start"] ? cfg["tracking"]["start"].as<int>() : 0;
        t_end_ = cfg["tracking"]["end"] ? cfg["tracking"]["end"].as<int>() : static_cast<int>(tracking_ctx_.size());
        t_stop_ = cfg["tracking"]["stop"] ? cfg["tracking"]["stop"].as<int>() : 0;
        gamma_ = cfg["tracking"]["gamma"] ? cfg["tracking"]["gamma"].as<float>() : 0.8f;
        window_size_ = cfg["tracking"]["window_size"] ? cfg["tracking"]["window_size"].as<int>() : 3;

        t_ = t_stop_;
        start_motion_ = false;

        spdlog::info("State_BFM(tracking): ctx={}, latent_dim={}, window={} gamma={}",
                     tracking_ctx_.size(), latent_dim_, window_size_, gamma_);
    }
    else if (task_type_ == TaskType::Reward)
    {
        if (cfg["reward"] && cfg["reward"]["selected_rewards_filter_z"])
        {
            auto selectors = cfg["reward"]["selected_rewards_filter_z"];
            for (std::size_t i = 0; i < selectors.size(); ++i)
            {
                auto item = selectors[i];
                std::string reward_name = item["reward"].as<std::string>();
                if (!npz.count(reward_name))
                {
                    spdlog::warn("State_BFM(reward): reward key not found in npz: {}", reward_name);
                    continue;
                }

                const auto arr_vec = to_float_vector(npz[reward_name]);
                const auto shape = npz[reward_name].shape;
                if (shape.size() != 2 || shape[1] == 0)
                {
                    spdlog::warn("State_BFM(reward): skip malformed key {}", reward_name);
                    continue;
                }
                latent_dim_ = shape[1];

                auto z_ids = item["z_ids"].as<std::vector<int>>();
                for (int z_id : z_ids)
                {
                    if (z_id < 0 || static_cast<std::size_t>(z_id) >= shape[0])
                    {
                        spdlog::warn("State_BFM(reward): invalid z_id {} for key {}", z_id, reward_name);
                        continue;
                    }
                    std::vector<float> z(latent_dim_, 0.0f);
                    std::copy(arr_vec.begin() + z_id * latent_dim_, arr_vec.begin() + (z_id + 1) * latent_dim_, z.begin());
                    selected_latents_.push_back(std::move(z));
                    selected_latent_names_.push_back(reward_name + "::z" + std::to_string(z_id));
                }
            }
        }
        else
        {
            // fallback: first z of every reward key
            for (const auto& kv : npz)
            {
                const auto shape = kv.second.shape;
                if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0)
                {
                    continue;
                }
                latent_dim_ = shape[1];
                const auto arr_vec = to_float_vector(kv.second);
                std::vector<float> z(latent_dim_, 0.0f);
                std::copy(arr_vec.begin(), arr_vec.begin() + latent_dim_, z.begin());
                selected_latents_.push_back(std::move(z));
                selected_latent_names_.push_back(kv.first + "::z0");
            }
        }

        if (selected_latents_.empty())
        {
            throw std::runtime_error("State_BFM(reward): no valid selected latent vectors");
        }
        spdlog::info("State_BFM(reward): selected {} latents", selected_latents_.size());
    }
    else
    {
        std::vector<std::string> selected_goals;
        if (cfg["goal"] && cfg["goal"]["selected_goals"])
        {
            selected_goals = cfg["goal"]["selected_goals"].as<std::vector<std::string>>();
        }

        if (selected_goals.empty())
        {
            for (const auto& kv : npz)
            {
                selected_goals.push_back(kv.first);
            }
        }

        for (const auto& goal_name : selected_goals)
        {
            if (!npz.count(goal_name))
            {
                spdlog::warn("State_BFM(goal): goal key not found in npz: {}", goal_name);
                continue;
            }
            const auto shape = npz[goal_name].shape;
            if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0)
            {
                spdlog::warn("State_BFM(goal): malformed goal key: {}", goal_name);
                continue;
            }
            latent_dim_ = shape[1];

            const auto arr_vec = to_float_vector(npz[goal_name]);
            std::vector<float> z(latent_dim_, 0.0f);
            std::copy(arr_vec.begin(), arr_vec.begin() + latent_dim_, z.begin());
            selected_latents_.push_back(std::move(z));
            selected_latent_names_.push_back(goal_name);
        }

        if (selected_latents_.empty())
        {
            throw std::runtime_error("State_BFM(goal): no valid selected goals");
        }
        spdlog::info("State_BFM(goal): selected {} goals", selected_latents_.size());
    }

    z_index_ = 0;
}

void State_BFM::load_key_config(const YAML::Node& cfg)
{
    auto gamepad_map = cfg["gamepad_map"];
    if (gamepad_map)
    {
        if (gamepad_map["start_motion"]) key_cfg_.start_motion = gamepad_map["start_motion"].as<std::string>();
        if (gamepad_map["next_latent"]) key_cfg_.next_latent = gamepad_map["next_latent"].as<std::string>();
        if (gamepad_map["reset_state"]) key_cfg_.reset_state = gamepad_map["reset_state"].as<std::string>();
    }

    auto compile = [](const std::string& expr) {
        unitree::common::dsl::Parser p(expr);
        auto ast = p.Parse();
        return unitree::common::dsl::Compile(*ast);
    };

    start_motion_trigger_ = compile(key_cfg_.start_motion);
    next_latent_trigger_ = compile(key_cfg_.next_latent);
    reset_state_trigger_ = compile(key_cfg_.reset_state);
}

void State_BFM::enter()
{
    if (!env_) {
        spdlog::warn("State_BFM::enter: env is null.");
        return;
    }

    // gains from deploy.yaml
    for (std::size_t i = 0; i < env_->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env_->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env_->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    env_->robot->update();
    const std::size_t dof = env_->robot->data.joint_ids_map.size();
    latest_q_target_.assign(dof, 0.0f);
    for (std::size_t i = 0; i < dof; ++i)
    {
        latest_q_target_[i] = env_->robot->data.joint_pos[i];
    }

    use_policy_action_ = true;
    start_motion_ = false;
    total_steps_ = 0;
    last_action_.assign(dof, 0.0f);
    t_ = t_stop_;
    z_index_ = 0;

    enter_time_ = std::chrono::steady_clock::now();

    policy_thread_running_ = true;
    policy_thread_ = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desired(env_->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desired);
        auto sleep_till = clock::now() + dt;
        double infer_time_ms_sum = 0.0;
        std::size_t infer_time_count = 0;

        while (policy_thread_running_)
        {
            env_->robot->update();

            auto input = build_policy_input();
            const auto infer_t0 = clock::now();
            auto action = infer_action(input);
            const auto infer_t1 = clock::now();
            infer_time_ms_sum += std::chrono::duration<double, std::milli>(infer_t1 - infer_t0).count();
            ++infer_time_count;

            for (auto& a : action)
            {
                a = std::clamp(a, -1.0f, 1.0f) * action_rescale_;
            }
            last_action_ = action;

            auto q_target = compute_q_target(action);
            q_target = clamp_vec(q_target, joint_pos_lower_limit_, joint_pos_upper_limit_);

            {
                std::lock_guard<std::mutex> lock(target_mtx_);
                latest_q_target_ = std::move(q_target);
            }

            ++total_steps_;
            if (total_steps_ % 200 == 0)
            {
                const double avg_infer_ms = (infer_time_count > 0) ? (infer_time_ms_sum / static_cast<double>(infer_time_count)) : 0.0;
                spdlog::info("State_BFM step={} latent={}/{} avg_infer_ms={:.3f}",
                             total_steps_,
                             (selected_latents_.empty() ? 0 : z_index_ + 1),
                             selected_latents_.size(),
                             avg_infer_ms);
                infer_time_ms_sum = 0.0;
                infer_time_count = 0;
            }

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    });
}

void State_BFM::run()
{
    if (!env_) return;

    handle_gamepad_events();

    std::vector<float> q_target;
    {
        std::lock_guard<std::mutex> lock(target_mtx_);
        q_target = latest_q_target_;
    }

    for (std::size_t i = 0; i < env_->robot->data.joint_ids_map.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[env_->robot->data.joint_ids_map[i]].q() = q_target[i];
    }
}

void State_BFM::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable())
    {
        policy_thread_.join();
    }
}

void State_BFM::handle_gamepad_events()
{
    if (!lowstate)
    {
        return;
    }

    const auto& joy = lowstate->joystick;

    if (start_motion_trigger_ && start_motion_trigger_(joy))
    {
        if (task_type_ == TaskType::Tracking)
        {
            start_motion_ = true;
            t_ = t_start_;
            spdlog::info("State_BFM(tracking): start motion from t={}", t_);
        }
        else
        {
            // Keep parity with python "start" semantics for non-tracking tasks.
            use_policy_action_ = true;
            spdlog::info("State_BFM: start trigger received");
        }
    }
    else if (next_latent_trigger_ && next_latent_trigger_(joy))
    {
        if (task_type_ == TaskType::Tracking)
        {
            if (!tracking_ctx_.empty())
            {
                t_ = (t_ + 1) % static_cast<int>(tracking_ctx_.size());
                spdlog::info("State_BFM(tracking): next latent frame t={}", t_);
            }
        }
        else if (!selected_latents_.empty())
        {
            z_index_ = (z_index_ + 1) % selected_latents_.size();
            spdlog::info("State_BFM: switch latent to {} ({}/{})", selected_latent_names_[z_index_], z_index_ + 1, selected_latents_.size());
        }
    }
    else if (reset_state_trigger_ && reset_state_trigger_(joy))
    {
        z_index_ = 0;
        start_motion_ = false;
        if (task_type_ == TaskType::Tracking)
        {
            t_ = t_stop_;
        }
        spdlog::info("State_BFM: reset stop state");
    }
}

std::vector<float> State_BFM::build_policy_obs() const
{
    auto obs_map = env_->observation_manager->compute();
    auto it_base = obs_map.find("obs_base");
    if (it_base == obs_map.end())
    {
        throw std::runtime_error("State_BFM: observation group 'obs_base' not found");
    }
    auto it_hist = obs_map.find("obs_hist");
    if (it_hist == obs_map.end())
    {
        throw std::runtime_error("State_BFM: observation group 'obs_hist' not found");
    }

    std::vector<float> obs = it_base->second;
    auto hist = it_hist->second;

    // ObservationManager history is oldest->newest for each term.
    // Convert to newest->oldest to match BFM python history layout.
    const std::size_t dof = env_->robot->data.joint_pos.size();
    const std::size_t hist_len = 4;
    const std::array<std::size_t, 5> dims = {dof, 3, dof, dof, 3};
    std::size_t cursor = 0;
    for (auto dim : dims)
    {
        const std::size_t block_size = dim * hist_len;
        if (cursor + block_size > hist.size())
        {
            throw std::runtime_error("State_BFM: obs_hist size mismatch");
        }
        for (std::size_t h = 0; h < hist_len; ++h)
        {
            const std::size_t src_h = hist_len - 1 - h;
            const std::size_t src = cursor + src_h * dim;
            obs.insert(obs.end(), hist.begin() + src, hist.begin() + src + dim);
        }
        cursor += block_size;
    }

    return obs;
}

std::vector<float> State_BFM::build_policy_input()
{
    // Keep `last_action` observation synchronized with BFM action buffer.
    env_->action_manager->process_action(last_action_);

    auto obs = build_policy_obs();
    std::vector<float> latent;
    if (task_type_ == TaskType::Tracking)
    {
        latent = compute_tracking_latent();
    }
    else
    {
        latent = get_current_latent();
    }

    std::vector<float> input;
    input.reserve(obs.size() + latent.size());
    input.insert(input.end(), obs.begin(), obs.end());
    input.insert(input.end(), latent.begin(), latent.end());
    return input;
}

std::vector<float> State_BFM::compute_tracking_latent()
{
    if (tracking_ctx_.empty())
    {
        throw std::runtime_error("State_BFM(tracking): empty tracking context");
    }

    int t_clamped = std::clamp(t_, 0, static_cast<int>(tracking_ctx_.size()) - 1);
    const int window_end = std::min<int>(t_clamped + std::max(window_size_, 1), static_cast<int>(tracking_ctx_.size()));
    const int n = std::max(1, window_end - t_clamped);

    std::vector<float> latent(latent_dim_, 0.0f);
    std::vector<float> discounts(n, 1.0f);
    for (int i = 0; i < n; ++i)
    {
        discounts[i] = std::pow(gamma_, static_cast<float>(i));
    }

    const float wsum = std::accumulate(discounts.begin(), discounts.end(), 0.0f);
    if (wsum > 1e-6f)
    {
        for (auto& d : discounts) d /= wsum;
    }

    for (int i = 0; i < n; ++i)
    {
        const auto& v = tracking_ctx_[t_clamped + i];
        for (std::size_t j = 0; j < latent_dim_; ++j)
        {
            latent[j] += discounts[i] * v[j];
        }
    }

    const float norm_latent = vec_norm(latent);
    const float norm_ref = vec_norm(tracking_ctx_[0]);
    if (norm_latent > 1e-6f)
    {
        const float scale = norm_ref / norm_latent;
        for (auto& x : latent) x *= scale;
    }

    if (use_policy_action_)
    {
        if (start_motion_ && t_ < t_end_)
        {
            ++t_;
            t_ %= static_cast<int>(tracking_ctx_.size());
        }
        else
        {
            t_ = t_stop_;
            start_motion_ = false;
        }
    }

    return latent;
}

std::vector<float> State_BFM::get_current_latent() const
{
    if (selected_latents_.empty())
    {
        throw std::runtime_error("State_BFM: selected_latents is empty");
    }
    return selected_latents_[z_index_];
}

std::vector<float> State_BFM::infer_action(const std::vector<float>& input)
{
    const std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(input.size())};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem,
        const_cast<float*>(input.data()),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_name_cstr = onnx_input_name_.c_str();
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name_cstr,
        &input_tensor,
        1,
        output_names_.data(),
        output_names_.size()
    );

    std::size_t action_out_idx = 0;
    for (std::size_t i = 0; i < output_names_str_.size(); ++i)
    {
        if (output_names_str_[i] == action_output_name_)
        {
            action_out_idx = i;
            break;
        }
    }

    auto& out = output_tensors[action_out_idx];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    std::size_t count = 1;
    for (auto d : shape)
    {
        count *= static_cast<std::size_t>(std::max<int64_t>(d, 1));
    }

    auto* out_ptr = out.GetTensorMutableData<float>();
    std::vector<float> action(out_ptr, out_ptr + count);

    if (action.size() != action_scale_.size())
    {
        const std::size_t keep = std::min(action.size(), action_scale_.size());
        action.resize(keep, 0.0f);
    }

    return action;
}

std::vector<float> State_BFM::compute_q_target(const std::vector<float>& scaled_action)
{
    const std::size_t dof = env_->robot->data.joint_pos.size();

    if (!use_policy_action_)
    {
        std::vector<float> q(dof, 0.0f);
        for (std::size_t i = 0; i < dof; ++i)
        {
            q[i] = env_->robot->data.joint_pos[i];
        }
        return q;
    }

    std::vector<float> q(dof, 0.0f);
    for (std::size_t i = 0; i < dof; ++i)
    {
        q[i] = default_joint_pos_[i] + scaled_action[i] * action_scale_[i];
    }
    return q;
}

std::vector<float> State_BFM::to_float_vector(const cnpy::NpyArray& array)
{
    if (array.word_size != sizeof(float))
    {
        throw std::runtime_error("State_BFM: only float32 npz arrays are supported");
    }

    const std::size_t n = array.num_vals;
    const auto* ptr = array.data<float>();
    return std::vector<float>(ptr, ptr + n);
}
