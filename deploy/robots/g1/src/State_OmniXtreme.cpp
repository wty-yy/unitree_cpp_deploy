#include "State_OmniXtreme.h"

#include "cnpy.h"
#include "unitree_articulation.h"
#include "unitree_joystick_dsl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace
{
enum class OnnxExecProvider
{
    TensorRT,
    CUDA,
    CPU
};

constexpr std::array<int, 29> kPerm = {
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
};

constexpr std::array<int, 29> kInvPerm = {
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
};

OnnxExecProvider append_best_provider(
    Ort::SessionOptions& options,
    int device_id,
    bool enable_tensorrt,
    bool enable_cuda,
    bool trt_fp16_enable,
    const std::string& trt_cache_path)
{
    const OrtApi& api = Ort::GetApi();

    char** providers = nullptr;
    int provider_count = 0;
    OrtStatus* status = api.GetAvailableProviders(&providers, &provider_count);
    if (status != nullptr)
    {
        spdlog::warn("State_OmniXtreme: GetAvailableProviders failed: {}", api.GetErrorMessage(status));
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
        trt_options.trt_max_partition_iterations = 1000;
        trt_options.trt_min_subgraph_size = 1;
        trt_options.trt_fp16_enable = trt_fp16_enable ? 1 : 0;
        trt_options.trt_engine_cache_enable = trt_cache_path.empty() ? 0 : 1;
        trt_options.trt_engine_cache_path = trt_cache_path.empty() ? nullptr : trt_cache_path.c_str();
        status = api.SessionOptionsAppendExecutionProvider_TensorRT(options, &trt_options);
        if (status == nullptr)
        {
            return OnnxExecProvider::TensorRT;
        }
        spdlog::warn("State_OmniXtreme: append TensorRT provider failed: {}", api.GetErrorMessage(status));
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
        spdlog::warn("State_OmniXtreme: append CUDA provider failed: {}", api.GetErrorMessage(status));
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

Eigen::Quaternionf xyzw_to_eigen_quat(const float* data)
{
    Eigen::Quaternionf q(data[3], data[0], data[1], data[2]);
    q.normalize();
    return q;
}

Eigen::Quaternionf wxyz_to_eigen_quat(const float* data)
{
    Eigen::Quaternionf q(data[0], data[1], data[2], data[3]);
    q.normalize();
    return q;
}

std::array<float, 6> quat_to_6d(const Eigen::Quaternionf& q)
{
    const Eigen::Matrix3f rot = q.toRotationMatrix();
    // Match Python: tmp_matrix[..., :2].reshape(-1, 6) in row-major order.
    return {rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1)};
}

float quat_to_yaw(const Eigen::Quaternionf& q)
{
    const float w = q.w();
    const float x = q.x();
    const float y = q.y();
    const float z = q.z();
    const float siny_cosp = 2.0f * (w * z + x * y);
    const float cosy_cosp = w * w + x * x - y * y - z * z;
    return std::atan2(siny_cosp, cosy_cosp);
}

std::vector<float> require_vec(const YAML::Node& node, const char* key, std::size_t expected_size)
{
    if (!node || !node[key])
    {
        throw std::runtime_error(std::string("State_OmniXtreme: missing omnixtreme.") + key + " in deploy.yaml");
    }
    auto values = node[key].as<std::vector<float>>();
    if (values.size() != expected_size)
    {
        throw std::runtime_error(std::string("State_OmniXtreme: omnixtreme.") + key + " size mismatch");
    }
    return values;
}

Eigen::Quaternionf quat_with_replaced_yaw(const Eigen::Quaternionf& q, float yaw)
{
    const float w = q.w();
    const float x = q.x();
    const float y = q.y();
    const float z = q.z();
    const float sinr_cosp = 2.0f * (w * x + y * z);
    const float cosr_cosp = w * w - x * x - y * y + z * z;
    const float roll = std::atan2(sinr_cosp, cosr_cosp);
    const float sinp = 2.0f * (w * y - z * x);
    const float pitch = std::asin(std::clamp(sinp, -1.0f, 1.0f));

    Eigen::AngleAxisf roll_angle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch_angle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yaw_angle(yaw, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf out = yaw_angle * pitch_angle * roll_angle;
    out.normalize();
    return out;
}

std::size_t product_of_shape(const std::vector<std::size_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<std::size_t>());
}

} // namespace

State_OmniXtreme::State_OmniXtreme(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    if (!cfg)
    {
        throw std::runtime_error("State_OmniXtreme: missing config for state " + state_string);
    }

    load_policy_and_env(cfg);
    initialize_limits(cfg);
    load_motion_library(cfg);
    load_key_config(cfg);
}

void State_OmniXtreme::load_policy_and_env(const YAML::Node& cfg)
{
    using clock = std::chrono::steady_clock;

    if (!cfg["policy_dir"] || cfg["policy_dir"].IsNull())
    {
        throw std::runtime_error("State_OmniXtreme: policy_dir is required");
    }

    policy_dir_ = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    const auto deploy_rel = cfg["deploy_yaml"] ? cfg["deploy_yaml"].as<std::string>() : "params/deploy.yaml";
    const auto base_model_rel = cfg["base_model"] ? cfg["base_model"].as<std::string>() : "exported/base_policy_trt.onnx";
    const auto residual_model_rel = cfg["residual_model"] ? cfg["residual_model"].as<std::string>() : "exported/residual_policy.onnx";
    const auto fk_model_rel = cfg["fk_model"] ? cfg["fk_model"].as<std::string>() : "exported/fk_trt.onnx";

    auto deploy_cfg = YAML::LoadFile((policy_dir_ / deploy_rel).string());
    deploy_cfg_ = deploy_cfg;
    env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );

    dof_ = env_->robot->data.joint_ids_map.size();
    const YAML::Node omni_cfg = deploy_cfg["omnixtreme"];
    action_scale_ = require_vec(omni_cfg, "action_scale", dof_);
    obs_default_joint_pos_ = require_vec(omni_cfg, "pd_bias_joint_pos", dof_);

    const bool prefer_cuda = cfg["onnx_cuda"] ? cfg["onnx_cuda"].as<bool>() : true;
    const bool prefer_tensorrt = cfg["onnx_tensorrt"] ? cfg["onnx_tensorrt"].as<bool>() : true;
    const int cuda_device_id = cfg["onnx_cuda_device"] ? cfg["onnx_cuda_device"].as<int>() : 0;
    const bool trt_fp16_enable = cfg["onnx_trt_fp16"] ? cfg["onnx_trt_fp16"].as<bool>() : true;
    const std::string trt_cache_dir =
        cfg["onnx_trt_cache_dir"] ? cfg["onnx_trt_cache_dir"].as<std::string>() : "trt_cache";
    const std::filesystem::path trt_cache_path =
        trt_cache_dir.empty() ? std::filesystem::path() : (policy_dir_ / trt_cache_dir);
    if (!trt_cache_path.empty())
    {
        std::filesystem::create_directories(trt_cache_path);
    }

    base_session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    residual_session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    const OnnxExecProvider base_provider = append_best_provider(
        base_session_options_, cuda_device_id, prefer_tensorrt, prefer_cuda, trt_fp16_enable, trt_cache_path.string());
    const OnnxExecProvider residual_provider = append_best_provider(
        residual_session_options_, cuda_device_id, prefer_tensorrt, prefer_cuda, trt_fp16_enable, trt_cache_path.string());
    Ort::SessionOptions fk_session_options;
    fk_session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    const OnnxExecProvider fk_provider = append_best_provider(
        fk_session_options, cuda_device_id, prefer_tensorrt, prefer_cuda, trt_fp16_enable, trt_cache_path.string());

    const auto base_model_path = (policy_dir_ / base_model_rel).string();
    const auto residual_model_path = (policy_dir_ / residual_model_rel).string();
    const auto fk_model_path = (policy_dir_ / fk_model_rel).string();

    spdlog::info("State_OmniXtreme: loading base model {}", base_model_path);
    auto t0 = clock::now();
    base_session_ = std::make_unique<Ort::Session>(ort_env_, base_model_path.c_str(), base_session_options_);
    auto t1 = clock::now();
    spdlog::info(
        "State_OmniXtreme: base model ready in {:.3f}s",
        std::chrono::duration<double>(t1 - t0).count());

    spdlog::info("State_OmniXtreme: loading residual model {}", residual_model_path);
    t0 = clock::now();
    residual_session_ = std::make_unique<Ort::Session>(ort_env_, residual_model_path.c_str(), residual_session_options_);
    t1 = clock::now();
    spdlog::info(
        "State_OmniXtreme: residual model ready in {:.3f}s",
        std::chrono::duration<double>(t1 - t0).count());

    spdlog::info("State_OmniXtreme: loading fk model {}", fk_model_path);
    t0 = clock::now();
    fk_session_ = std::make_unique<Ort::Session>(ort_env_, fk_model_path.c_str(), fk_session_options);
    t1 = clock::now();
    spdlog::info(
        "State_OmniXtreme: fk model ready in {:.3f}s",
        std::chrono::duration<double>(t1 - t0).count());

    for (std::size_t i = 0; i < base_session_->GetInputCount(); ++i)
    {
        auto name = base_session_->GetInputNameAllocated(i, allocator_);
        base_input_names_str_.push_back(name.get());
        auto input_type = base_session_->GetInputTypeInfo(i);
        base_input_shapes_.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
    }
    base_input_names_.clear();
    base_input_names_.reserve(base_input_names_str_.size());
    for (const auto& name : base_input_names_str_)
    {
        base_input_names_.push_back(name.c_str());
    }

    for (std::size_t i = 0; i < base_session_->GetOutputCount(); ++i)
    {
        auto name = base_session_->GetOutputNameAllocated(i, allocator_);
        base_output_names_str_.push_back(name.get());
    }
    base_output_names_.clear();
    base_output_names_.reserve(base_output_names_str_.size());
    for (const auto& name : base_output_names_str_)
    {
        base_output_names_.push_back(name.c_str());
    }
    if (!base_output_names_str_.empty() &&
        std::find(base_output_names_str_.begin(), base_output_names_str_.end(), base_action_output_name_) == base_output_names_str_.end())
    {
        base_action_output_name_ = base_output_names_str_.front();
    }

    auto residual_input_name = residual_session_->GetInputNameAllocated(0, allocator_);
    residual_input_name_ = residual_input_name.get();
    for (std::size_t i = 0; i < residual_session_->GetOutputCount(); ++i)
    {
        auto name = residual_session_->GetOutputNameAllocated(i, allocator_);
        residual_output_names_str_.push_back(name.get());
    }
    residual_output_names_.clear();
    residual_output_names_.reserve(residual_output_names_str_.size());
    for (const auto& name : residual_output_names_str_)
    {
        residual_output_names_.push_back(name.c_str());
    }
    if (!residual_output_names_str_.empty() &&
        std::find(residual_output_names_str_.begin(), residual_output_names_str_.end(), residual_action_output_name_) == residual_output_names_str_.end())
    {
        residual_action_output_name_ = residual_output_names_str_.front();
    }

    auto log_provider = [](const char* model_name, OnnxExecProvider provider) {
        if (provider == OnnxExecProvider::TensorRT)
        {
            spdlog::info("State_OmniXtreme: {} execution provider: TensorRT", model_name);
        }
        else if (provider == OnnxExecProvider::CUDA)
        {
            spdlog::info("State_OmniXtreme: {} execution provider: CUDA", model_name);
        }
        else
        {
            spdlog::warn("State_OmniXtreme: {} execution provider fallback to CPU", model_name);
        }
    };
    log_provider("base model", base_provider);
    log_provider("residual model", residual_provider);
    log_provider("fk model", fk_provider);

    for (std::size_t i = 0; i < fk_session_->GetOutputCount(); ++i)
    {
        auto name = fk_session_->GetOutputNameAllocated(i, allocator_);
        fk_output_names_str_.push_back(name.get());
    }
    fk_output_names_.clear();
    fk_output_names_.reserve(fk_output_names_str_.size());
    for (const auto& name : fk_output_names_str_)
    {
        fk_output_names_.push_back(name.c_str());
    }

    for (std::size_t i = 0; i < base_input_names_str_.size(); ++i)
    {
        const auto& name = base_input_names_str_[i];
        if (name == "real_obs" && base_input_shapes_[i].size() >= 2)
        {
            real_obs_dim_ = static_cast<std::size_t>(base_input_shapes_[i][1]);
        }
        else if (name == "command_obs" && base_input_shapes_[i].size() >= 2)
        {
            command_obs_dim_ = static_cast<std::size_t>(base_input_shapes_[i][1]);
        }
        else if (name == "real_historical_obs_raw" && base_input_shapes_[i].size() >= 2)
        {
            history_obs_dim_ = static_cast<std::size_t>(base_input_shapes_[i][1]);
        }
    }
    history_length_ = std::max<std::size_t>(1, history_obs_dim_ / std::max<std::size_t>(1, real_obs_dim_));
    residual_obs_dim_ = real_obs_dim_ + command_obs_dim_ + dof_;
}

void State_OmniXtreme::initialize_limits(const YAML::Node& cfg)
{
    joint_pos_lower_limit_.assign(dof_, -std::numeric_limits<float>::max());
    joint_pos_upper_limit_.assign(dof_, std::numeric_limits<float>::max());

    if (cfg["joint_pos_lower_limit"])
    {
        joint_pos_lower_limit_ = cfg["joint_pos_lower_limit"].as<std::vector<float>>();
    }
    if (cfg["joint_pos_upper_limit"])
    {
        joint_pos_upper_limit_ = cfg["joint_pos_upper_limit"].as<std::vector<float>>();
    }
    if (joint_pos_lower_limit_.size() != dof_ || joint_pos_upper_limit_.size() != dof_)
    {
        throw std::runtime_error("State_OmniXtreme: joint position limit size mismatch");
    }

    action_clip_ = cfg["action_clip"] ? cfg["action_clip"].as<float>() : 1.0f;
    residual_scale_ = cfg["residual_scale"] ? cfg["residual_scale"].as<float>() : 1.0f;
    q_target_lpf_alpha_ = cfg["q_target_lpf_alpha"] ? cfg["q_target_lpf_alpha"].as<float>() : 0.8f;
    tau_ff_lpf_alpha_ = cfg["tau_ff_lpf_alpha"] ? cfg["tau_ff_lpf_alpha"].as<float>() : 0.8f;
    waist_q_target_lpf_alpha_ =
        cfg["waist_q_target_lpf_alpha"] ? cfg["waist_q_target_lpf_alpha"].as<float>() : q_target_lpf_alpha_;
    waist_tau_ff_lpf_alpha_ =
        cfg["waist_tau_ff_lpf_alpha"] ? cfg["waist_tau_ff_lpf_alpha"].as<float>() : 0.5f;
    q_target_lpf_alpha_ = std::clamp(q_target_lpf_alpha_, 0.0f, 1.0f);
    tau_ff_lpf_alpha_ = std::clamp(tau_ff_lpf_alpha_, 0.0f, 1.0f);
    waist_q_target_lpf_alpha_ = std::clamp(waist_q_target_lpf_alpha_, 0.0f, 1.0f);
    waist_tau_ff_lpf_alpha_ = std::clamp(waist_tau_ff_lpf_alpha_, 0.0f, 1.0f);
    loop_trajectory_ = cfg["loop_trajectory"] ? cfg["loop_trajectory"].as<bool>() : true;
    const YAML::Node omni_cfg = deploy_cfg_["omnixtreme"];
    p_gains_ = require_vec(omni_cfg, "p_gains", dof_);
    d_gains_ = require_vec(omni_cfg, "d_gains", dof_);
    x1_ = require_vec(omni_cfg, "envelope_x1", dof_);
    x2_ = require_vec(omni_cfg, "envelope_x2", dof_);
    y1_ = require_vec(omni_cfg, "envelope_y1", dof_);
    y2_ = require_vec(omni_cfg, "envelope_y2", dof_);
    va_ = require_vec(omni_cfg, "friction_va", dof_);
    fs_ = require_vec(omni_cfg, "friction_fs", dof_);
    fd_ = require_vec(omni_cfg, "friction_fd", dof_);
    last_action_.assign(dof_, 0.0f);
    last_base_action_.assign(dof_, 0.0f);
    latest_q_target_.assign(dof_, 0.0f);
    latest_tau_ff_.assign(dof_, 0.0f);
}

void State_OmniXtreme::load_motion_library(const YAML::Node& cfg)
{
    const auto motions_rel = cfg["motions_dir"] ? cfg["motions_dir"].as<std::string>() : "exported/motions";
    const auto motions_dir = policy_dir_ / motions_rel;
    if (cfg["root_body_index"]) root_body_index_ = cfg["root_body_index"].as<int>();
    if (cfg["anchor_body_index"]) anchor_body_index_ = cfg["anchor_body_index"].as<int>();

    std::vector<std::filesystem::path> motion_files;
    if (cfg["motion_files"] && cfg["motion_files"].IsSequence())
    {
        const auto configured_files = cfg["motion_files"].as<std::vector<std::string>>();
        motion_files.reserve(configured_files.size());
        for (const auto& rel_path : configured_files)
        {
            const auto path = policy_dir_ / rel_path;
            if (!std::filesystem::exists(path))
            {
                throw std::runtime_error("State_OmniXtreme: motion file not found: " + path.string());
            }
            motion_files.push_back(path);
        }
    }
    else
    {
        for (const auto& entry : std::filesystem::directory_iterator(motions_dir))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".npz")
            {
                motion_files.push_back(entry.path());
            }
        }
    }

    std::sort(motion_files.begin(), motion_files.end(), [](const auto& a, const auto& b) {
        const auto sa = a.stem().string();
        const auto sb = b.stem().string();
        try
        {
            return std::stoi(sa) < std::stoi(sb);
        }
        catch (...)
        {
            return sa < sb;
        }
    });

    trajectories_.clear();
    for (const auto& motion_file : motion_files)
    {
        auto npz = cnpy::npz_load(motion_file.string());
        if (!npz.count("joint_pos") || !npz.count("joint_vel") || !npz.count("body_quat_w"))
        {
            spdlog::warn("State_OmniXtreme: skip incomplete motion file {}", motion_file.string());
            continue;
        }

        const auto joint_pos_shape = npz["joint_pos"].shape;
        const auto joint_vel_shape = npz["joint_vel"].shape;
        const auto body_quat_shape = npz["body_quat_w"].shape;
        if (joint_pos_shape.size() != 2 || joint_pos_shape[1] != dof_ ||
            joint_vel_shape.size() != 2 || joint_vel_shape[1] != dof_ ||
            body_quat_shape.size() != 3 || body_quat_shape[2] != 4)
        {
            spdlog::warn("State_OmniXtreme: skip malformed motion file {}", motion_file.string());
            continue;
        }

        const auto frame_count = joint_pos_shape[0];
        const auto body_count = body_quat_shape[1];
        if (root_body_index_ < 0 || static_cast<std::size_t>(root_body_index_) >= body_count ||
            anchor_body_index_ < 0 || static_cast<std::size_t>(anchor_body_index_) >= body_count)
        {
            throw std::runtime_error("State_OmniXtreme: invalid root/anchor body index");
        }

        const auto joint_pos = to_float_vector(npz["joint_pos"]);
        const auto joint_vel = to_float_vector(npz["joint_vel"]);
        const auto body_quat = to_float_vector(npz["body_quat_w"]);

        MotionTrajectory traj;
        traj.name = motion_file.stem().string();
        traj.ref_joint_pos_frames.reserve(frame_count);
        traj.ref_joint_vel_frames.reserve(frame_count);
        traj.ref_anchor_quat_frames.reserve(frame_count);
        traj.ref_root_quat_frames.reserve(frame_count);

        const std::size_t joint_stride = dof_;
        const std::size_t body_quat_stride = body_count * 4;
        for (std::size_t frame = 0; frame < frame_count; ++frame)
        {
            const float* joint_pos_ptr = joint_pos.data() + frame * joint_stride;
            const float* joint_vel_ptr = joint_vel.data() + frame * joint_stride;
            const float* root_quat_ptr = body_quat.data() + frame * body_quat_stride + root_body_index_ * 4;
            const float* anchor_quat_ptr = body_quat.data() + frame * body_quat_stride + anchor_body_index_ * 4;

            std::vector<float> ref_joint_pos(dof_, 0.0f);
            std::vector<float> ref_joint_vel(dof_, 0.0f);
            for (std::size_t i = 0; i < dof_; ++i)
            {
                ref_joint_pos[i] = joint_pos_ptr[kPerm[i]];
                ref_joint_vel[i] = joint_vel_ptr[kPerm[i]];
            }

            traj.ref_joint_pos_frames.push_back(std::move(ref_joint_pos));
            traj.ref_joint_vel_frames.push_back(std::move(ref_joint_vel));
            traj.ref_root_quat_frames.push_back(wxyz_to_eigen_quat(root_quat_ptr));
            traj.ref_anchor_quat_frames.push_back(wxyz_to_eigen_quat(anchor_quat_ptr));
        }

        trajectories_.push_back(std::move(traj));
    }

    if (trajectories_.empty())
    {
        throw std::runtime_error("State_OmniXtreme: no valid motion trajectories found");
    }

    trajectory_index_ = 0;
    frame_index_ = trajectories_[0].ref_joint_pos_frames.size() > 1 ? 1 : 0;
}

void State_OmniXtreme::load_key_config(const YAML::Node& cfg)
{
    auto gamepad_map = cfg["gamepad_map"];
    if (gamepad_map)
    {
        if (gamepad_map["next_trajectory"]) key_cfg_.next_trajectory = gamepad_map["next_trajectory"].as<std::string>();
        if (gamepad_map["previous_trajectory"]) key_cfg_.previous_trajectory = gamepad_map["previous_trajectory"].as<std::string>();
        if (gamepad_map["reset_trajectory"]) key_cfg_.reset_trajectory = gamepad_map["reset_trajectory"].as<std::string>();
        if (gamepad_map["toggle_execute"]) key_cfg_.toggle_execute = gamepad_map["toggle_execute"].as<std::string>();
    }

    auto compile = [](const std::string& expr) {
        unitree::common::dsl::Parser p(expr);
        auto ast = p.Parse();
        return unitree::common::dsl::Compile(*ast);
    };

    next_trajectory_trigger_ = compile(key_cfg_.next_trajectory);
    previous_trajectory_trigger_ = compile(key_cfg_.previous_trajectory);
    reset_trajectory_trigger_ = compile(key_cfg_.reset_trajectory);
    toggle_execute_trigger_ = compile(key_cfg_.toggle_execute);
}

void State_OmniXtreme::enter()
{
    if (!env_)
    {
        spdlog::warn("State_OmniXtreme::enter: env is null.");
        return;
    }

    for (std::size_t i = 0; i < dof_; ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = p_gains_[i];
        lowcmd->msg_.motor_cmd()[i].kd() = d_gains_[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0.0f;
        lowcmd->msg_.motor_cmd()[i].tau() = 0.0f;
    }

    env_->robot->update();
    latest_q_target_.assign(dof_, 0.0f);
    for (std::size_t i = 0; i < dof_; ++i)
    {
        latest_q_target_[i] = env_->robot->data.joint_pos[i];
    }

    total_steps_ = 0;
    reset_tracking_state(true);
    calibrate_yaw_alignment();
    execute_motion_ = false;
    warmup_models();
    spdlog::info("State_OmniXtreme: current trajectory {} ({}/{}) [paused]",
                 trajectories_[trajectory_index_].name, trajectory_index_ + 1, trajectories_.size());

    policy_thread_running_ = true;
    policy_thread_ = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const auto dt = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(env_->step_dt));
        auto sleep_till = clock::now() + dt;
        double fk_time_ms_sum = 0.0;
        double base_time_ms_sum = 0.0;
        double residual_time_ms_sum = 0.0;
        double step_time_ms_sum = 0.0;
        std::size_t timing_count = 0;

        while (policy_thread_running_)
        {
            const auto step_t0 = clock::now();
            env_->robot->update();

            const auto& traj = trajectories_[trajectory_index_];
            const std::size_t obs_frame_index = execute_motion_ ? frame_index_ : paused_frame_index();
            const auto fk_t0 = clock::now();
            auto command_obs = build_command_obs(traj, obs_frame_index);
            const auto fk_t1 = clock::now();
            auto real_obs = build_real_obs({});
            auto history_obs = build_history_obs(real_obs);
            const auto base_t0 = clock::now();
            auto base_action = infer_base_action(real_obs, command_obs, history_obs);
            const auto base_t1 = clock::now();
            auto residual_obs = build_residual_obs(real_obs, command_obs, base_action);
            const auto residual_t0 = clock::now();
            auto residual_action = infer_residual_action(residual_obs);
            const auto residual_t1 = clock::now();

            std::vector<float> final_action(dof_, 0.0f);
            for (std::size_t i = 0; i < dof_; ++i)
            {
                final_action[i] = base_action[i] + residual_scale_ * residual_action[i];
            }

            auto q_target = compute_q_target(final_action);
            std::vector<float> tau_ff(dof_, 0.0f);
            const float v_eps = 1e-2f;
            const auto& joint_pos = env_->robot->data.joint_pos;
            const auto& joint_vel = env_->robot->data.joint_vel;
            for (std::size_t i = 0; i < dof_; ++i)
            {
                const float dq = joint_vel[i];
                const float abs_dq = std::abs(dq);
                const float over = std::max(abs_dq - x1_[i], 0.0f);
                const float base_pos = abs_dq <= v_eps ? y2_[i] : (dq >= 0.0f ? y1_[i] : y2_[i]);
                const float base_neg = abs_dq <= v_eps ? -y2_[i] : (dq >= 0.0f ? -y2_[i] : -y1_[i]);
                const float denom = std::max(x2_[i] - x1_[i], 1e-6f);
                const float tau_high = std::max(base_pos - (base_pos / denom) * over, 0.0f);
                const float tau_low = std::min(base_neg + ((-base_neg) / denom) * over, 0.0f);

                const float p_limits_low = tau_low + d_gains_[i] * dq;
                const float p_limits_high = tau_high + d_gains_[i] * dq;
                const float q_low = p_limits_low / p_gains_[i] + joint_pos[i];
                const float q_high = p_limits_high / p_gains_[i] + joint_pos[i];
                q_target[i] = std::clamp(q_target[i], q_low, q_high);

                tau_ff[i] = -(fs_[i] * std::tanh(dq / va_[i]) + fd_[i] * dq);
            }
            {
                std::lock_guard<std::mutex> lock(target_mtx_);
                for (std::size_t i = 0; i < dof_; ++i)
                {
                    const bool is_waist = (i >= 12 && i <= 14);
                    const float q_alpha = is_waist ? waist_q_target_lpf_alpha_ : q_target_lpf_alpha_;
                    const float tau_alpha = is_waist ? waist_tau_ff_lpf_alpha_ : tau_ff_lpf_alpha_;
                    q_target[i] =
                        (1.0f - q_alpha) * latest_q_target_[i] + q_alpha * q_target[i];
                    tau_ff[i] =
                        (1.0f - tau_alpha) * latest_tau_ff_[i] + tau_alpha * tau_ff[i];
                }
                latest_q_target_ = std::move(q_target);
                latest_tau_ff_ = std::move(tau_ff);
            }

            fk_time_ms_sum += std::chrono::duration<double, std::milli>(fk_t1 - fk_t0).count();
            base_time_ms_sum += std::chrono::duration<double, std::milli>(base_t1 - base_t0).count();
            residual_time_ms_sum += std::chrono::duration<double, std::milli>(residual_t1 - residual_t0).count();
            step_time_ms_sum += std::chrono::duration<double, std::milli>(clock::now() - step_t0).count();
            ++timing_count;

            last_action_ = final_action;
            last_base_action_ = base_action;
            if (execute_motion_)
            {
                advance_frame();
                ++total_steps_;
            }

            if (timing_count % 200 == 0)
            {
                spdlog::info(
                    "State_OmniXtreme step={} traj={} running={} avg_fk_ms={:.3f} avg_base_ms={:.3f} avg_res_ms={:.3f} avg_step_ms={:.3f}",
                    total_steps_,
                    trajectories_[trajectory_index_].name,
                    execute_motion_.load(),
                    fk_time_ms_sum / static_cast<double>(timing_count),
                    base_time_ms_sum / static_cast<double>(timing_count),
                    residual_time_ms_sum / static_cast<double>(timing_count),
                    step_time_ms_sum / static_cast<double>(timing_count));
                fk_time_ms_sum = 0.0;
                base_time_ms_sum = 0.0;
                residual_time_ms_sum = 0.0;
                step_time_ms_sum = 0.0;
                timing_count = 0;
            }

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    });
}

void State_OmniXtreme::warmup_models()
{
    using clock = std::chrono::steady_clock;

    if (trajectories_.empty())
    {
        return;
    }

    const auto& traj = trajectories_[trajectory_index_];
    const std::size_t warmup_frame = paused_frame_index();
    constexpr int kWarmupIters = 4;

    spdlog::info("State_OmniXtreme: warmup start iterations={}", kWarmupIters);
    const auto warmup_t0 = clock::now();

    double fk_ms_sum = 0.0;
    double base_ms_sum = 0.0;
    double residual_ms_sum = 0.0;

    for (int i = 0; i < kWarmupIters; ++i)
    {
        env_->robot->update();

        const auto fk_t0 = clock::now();
        auto command_obs = build_command_obs(traj, warmup_frame);
        const auto fk_t1 = clock::now();

        auto real_obs = build_real_obs({});
        auto history_obs = build_history_obs(real_obs);

        const auto base_t0 = clock::now();
        auto base_action = infer_base_action(real_obs, command_obs, history_obs);
        const auto base_t1 = clock::now();

        auto residual_obs = build_residual_obs(real_obs, command_obs, base_action);
        const auto residual_t0 = clock::now();
        auto residual_action = infer_residual_action(residual_obs);
        const auto residual_t1 = clock::now();

        std::vector<float> final_action(dof_, 0.0f);
        for (std::size_t j = 0; j < dof_; ++j)
        {
            final_action[j] = base_action[j] + residual_scale_ * residual_action[j];
        }

        {
            std::lock_guard<std::mutex> lock(target_mtx_);
            latest_q_target_ = compute_q_target(final_action);
            latest_tau_ff_.assign(dof_, 0.0f);
        }
        last_action_ = std::move(final_action);
        last_base_action_ = std::move(base_action);

        fk_ms_sum += std::chrono::duration<double, std::milli>(fk_t1 - fk_t0).count();
        base_ms_sum += std::chrono::duration<double, std::milli>(base_t1 - base_t0).count();
        residual_ms_sum += std::chrono::duration<double, std::milli>(residual_t1 - residual_t0).count();
    }

    const auto warmup_t1 = clock::now();
    spdlog::info(
        "State_OmniXtreme: warmup done total_s={:.3f} avg_fk_ms={:.3f} avg_base_ms={:.3f} avg_res_ms={:.3f}",
        std::chrono::duration<double>(warmup_t1 - warmup_t0).count(),
        fk_ms_sum / static_cast<double>(kWarmupIters),
        base_ms_sum / static_cast<double>(kWarmupIters),
        residual_ms_sum / static_cast<double>(kWarmupIters));
}

void State_OmniXtreme::run()
{
    handle_gamepad_events();

    std::lock_guard<std::mutex> lock(target_mtx_);
    for (std::size_t i = 0; i < dof_; ++i)
    {
        lowcmd->msg_.motor_cmd()[env_->robot->data.joint_ids_map[i]].q() = latest_q_target_[i];
        lowcmd->msg_.motor_cmd()[env_->robot->data.joint_ids_map[i]].tau() = latest_tau_ff_[i];
    }
}

void State_OmniXtreme::calibrate_yaw_alignment()
{
    if (trajectories_.empty() || trajectories_[trajectory_index_].ref_root_quat_frames.empty())
    {
        initial_yaw_offset_ = 0.0f;
        return;
    }

    const Eigen::Quaternionf current_root_quat = env_->robot->data.root_quat_w.normalized();
    const Eigen::Quaternionf ref_root_quat = trajectories_[trajectory_index_].ref_root_quat_frames.front();
    const float current_yaw = quat_to_yaw(current_root_quat);
    const float ref_yaw = quat_to_yaw(ref_root_quat);
    initial_yaw_offset_ = current_yaw - ref_yaw;

    spdlog::info("State_OmniXtreme: calibrate yaw current={:.3f} ref={:.3f} offset={:.3f}",
                 current_yaw, ref_yaw, initial_yaw_offset_);
}

void State_OmniXtreme::exit()
{
    policy_thread_running_ = false;
    if (policy_thread_.joinable())
    {
        policy_thread_.join();
    }
}

std::vector<float> State_OmniXtreme::build_real_obs(const std::vector<float>&) const
{
    std::vector<float> obs;
    obs.reserve(real_obs_dim_);

    const auto& joint_pos = env_->robot->data.joint_pos;
    const auto& joint_vel = env_->robot->data.joint_vel;
    for (std::size_t i = 0; i < dof_; ++i)
    {
        obs.push_back(joint_pos[i] - obs_default_joint_pos_[i]);
    }
    obs.insert(obs.end(), joint_vel.data(), joint_vel.data() + joint_vel.size());
    obs.push_back(env_->robot->data.root_ang_vel_b.x());
    obs.push_back(env_->robot->data.root_ang_vel_b.y());
    obs.push_back(env_->robot->data.root_ang_vel_b.z());
    obs.insert(obs.end(), last_base_action_.begin(), last_base_action_.end());

    if (obs.size() != real_obs_dim_)
    {
        throw std::runtime_error("State_OmniXtreme: real_obs dim mismatch");
    }
    return obs;
}

std::vector<float> State_OmniXtreme::build_command_obs(const MotionTrajectory& traj, std::size_t frame_index)
{
    const auto idx = std::min(frame_index, traj.ref_joint_pos_frames.size() - 1);
    const auto& ref_joint_pos = traj.ref_joint_pos_frames[idx];
    const auto& ref_joint_vel = traj.ref_joint_vel_frames[idx];
    const auto& ref_anchor_quat = traj.ref_anchor_quat_frames[idx];

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::array<float, 3> joint_angles = {
        env_->robot->data.joint_pos[12],
        env_->robot->data.joint_pos[13],
        env_->robot->data.joint_pos[14],
    };
    std::array<float, 3> base_pos = {0.0f, 0.0f, 0.79f};
    const Eigen::Quaternionf adjusted_root_quat = quat_with_replaced_yaw(
        env_->robot->data.root_quat_w.normalized(),
        quat_to_yaw(env_->robot->data.root_quat_w.normalized()) - initial_yaw_offset_);
    std::array<float, 4> base_quat = {
        adjusted_root_quat.w(),
        adjusted_root_quat.x(),
        adjusted_root_quat.y(),
        adjusted_root_quat.z()
    };
    std::array<int64_t, 2> shape3 = {1, 3};
    std::array<int64_t, 2> shape4 = {1, 4};

    std::array<const char*, 3> fk_input_names = {"joint_angles", "base_pos", "base_quat"};
    std::array<Ort::Value, 3> fk_input_tensors = {
        Ort::Value::CreateTensor<float>(memory_info, joint_angles.data(), joint_angles.size(), shape3.data(), shape3.size()),
        Ort::Value::CreateTensor<float>(memory_info, base_pos.data(), base_pos.size(), shape3.data(), shape3.size()),
        Ort::Value::CreateTensor<float>(memory_info, base_quat.data(), base_quat.size(), shape4.data(), shape4.size())
    };

    auto fk_outputs = fk_session_->Run(
        Ort::RunOptions{nullptr},
        fk_input_names.data(),
        fk_input_tensors.data(),
        fk_input_tensors.size(),
        fk_output_names_.data(),
        fk_output_names_.size());

    if (fk_outputs.size() < 2)
    {
        throw std::runtime_error("State_OmniXtreme: invalid fk outputs");
    }

    const float* rot_data = fk_outputs[1].GetTensorMutableData<float>();
    Eigen::Quaternionf current_anchor_quat = wxyz_to_eigen_quat(rot_data);
    current_anchor_quat.normalize();
    Eigen::Quaternionf relative_quat = current_anchor_quat.conjugate() * ref_anchor_quat;
    relative_quat.normalize();
    const auto anchor_6d = quat_to_6d(relative_quat);

    std::vector<float> command_obs;
    command_obs.reserve(command_obs_dim_);
    command_obs.insert(command_obs.end(), ref_joint_pos.begin(), ref_joint_pos.end());
    command_obs.insert(command_obs.end(), ref_joint_vel.begin(), ref_joint_vel.end());
    command_obs.insert(command_obs.end(), anchor_6d.begin(), anchor_6d.end());
    return command_obs;
}

std::vector<float> State_OmniXtreme::build_residual_obs(
    const std::vector<float>& real_obs,
    const std::vector<float>& command_obs,
    const std::vector<float>& base_action) const
{
    std::vector<float> residual_obs;
    residual_obs.reserve(residual_obs_dim_);

    for (int idx : kInvPerm)
    {
        residual_obs.push_back(command_obs[idx]);
    }
    for (int idx : kInvPerm)
    {
        residual_obs.push_back(command_obs[dof_ + idx]);
    }
    residual_obs.insert(residual_obs.end(), command_obs.begin() + static_cast<long>(2 * dof_), command_obs.end());

    residual_obs.push_back(real_obs[2 * dof_ + 0]);
    residual_obs.push_back(real_obs[2 * dof_ + 1]);
    residual_obs.push_back(real_obs[2 * dof_ + 2]);
    for (int idx : kInvPerm)
    {
        residual_obs.push_back(real_obs[idx]);
    }
    for (int idx : kInvPerm)
    {
        residual_obs.push_back(real_obs[dof_ + idx]);
    }
    for (int idx : kInvPerm)
    {
        residual_obs.push_back(last_action_[idx]);
    }
    for (int idx : kInvPerm)
    {
        residual_obs.push_back(base_action[idx]);
    }

    return residual_obs;
}

std::vector<float> State_OmniXtreme::build_history_obs(const std::vector<float>& current_real_obs)
{
    if (real_obs_history_.empty())
    {
        for (std::size_t i = 0; i < history_length_; ++i)
        {
            real_obs_history_.push_back(std::vector<float>(real_obs_dim_, 0.0f));
        }
    }

    real_obs_history_.pop_front();
    real_obs_history_.push_back(current_real_obs);

    std::vector<float> history_obs;
    history_obs.reserve(history_obs_dim_);
    for (const auto& obs : real_obs_history_)
    {
        history_obs.insert(history_obs.end(), obs.begin(), obs.end());
    }
    return history_obs;
}

std::vector<float> State_OmniXtreme::infer_base_action(
    const std::vector<float>& real_obs,
    const std::vector<float>& command_obs,
    const std::vector<float>& real_hist)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    std::vector<std::vector<float>> input_buffers;
    std::vector<Ort::Value> input_tensors;
    input_buffers.reserve(base_input_names_str_.size());
    input_tensors.reserve(base_input_names_str_.size());

    for (std::size_t i = 0; i < base_input_names_str_.size(); ++i)
    {
        const auto& name = base_input_names_str_[i];
        if (name == "real_obs")
        {
            input_buffers.push_back(real_obs);
        }
        else if (name == "command_obs")
        {
            input_buffers.push_back(command_obs);
        }
        else if (name == "real_historical_obs_raw")
        {
            input_buffers.push_back(real_hist);
        }
        else if (name == "initial_noise")
        {
            const std::size_t dim = base_input_shapes_[i].size() >= 2 ? static_cast<std::size_t>(std::max<int64_t>(1, base_input_shapes_[i][1])) : dof_;
            input_buffers.emplace_back(dim, 0.0f);
            for (auto& v : input_buffers.back())
            {
                v = normal_dist_(rng_);
            }
        }
        else
        {
            const std::size_t dim = base_input_shapes_[i].size() >= 2 ? static_cast<std::size_t>(std::max<int64_t>(1, base_input_shapes_[i][1])) : 1;
            input_buffers.emplace_back(dim, 0.0f);
        }

        auto shape = base_input_shapes_[i];
        if (!shape.empty())
        {
            shape[0] = 1;
        }
        else
        {
            shape = {1, static_cast<int64_t>(input_buffers.back().size())};
        }

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_buffers.back().data(),
            input_buffers.back().size(),
            shape.data(),
            shape.size()));
    }

    auto output_tensors = base_session_->Run(
        Ort::RunOptions{nullptr},
        base_input_names_.data(),
        input_tensors.data(),
        input_tensors.size(),
        base_output_names_.data(),
        base_output_names_.size());

    for (std::size_t i = 0; i < base_output_names_str_.size(); ++i)
    {
        if (base_output_names_str_[i] == base_action_output_name_)
        {
            auto* data = output_tensors[i].GetTensorMutableData<float>();
            return std::vector<float>(data, data + dof_);
        }
    }

    auto* data = output_tensors.front().GetTensorMutableData<float>();
    return std::vector<float>(data, data + dof_);
}

std::vector<float> State_OmniXtreme::infer_residual_action(const std::vector<float>& residual_obs)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(residual_obs.size())};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(residual_obs.data()),
        residual_obs.size(),
        input_shape.data(),
        input_shape.size());

    const char* input_name = residual_input_name_.c_str();
    auto output_tensors = residual_session_->Run(
        Ort::RunOptions{nullptr},
        &input_name,
        &input_tensor,
        1,
        residual_output_names_.data(),
        residual_output_names_.size());

    for (std::size_t i = 0; i < residual_output_names_str_.size(); ++i)
    {
        if (residual_output_names_str_[i] == residual_action_output_name_)
        {
            auto* data = output_tensors[i].GetTensorMutableData<float>();
            std::vector<float> proto(data, data + dof_);
            std::vector<float> out(dof_, 0.0f);
            for (std::size_t j = 0; j < dof_; ++j)
            {
                out[j] = proto[kPerm[j]];
            }
            return out;
        }
    }

    auto* data = output_tensors.front().GetTensorMutableData<float>();
    std::vector<float> proto(data, data + dof_);
    std::vector<float> out(dof_, 0.0f);
    for (std::size_t j = 0; j < dof_; ++j)
    {
        out[j] = proto[kPerm[j]];
    }
    return out;
}

std::vector<float> State_OmniXtreme::compute_q_target(const std::vector<float>& action) const
{
    std::vector<float> q(dof_, 0.0f);
    for (std::size_t i = 0; i < dof_; ++i)
    {
        q[i] = obs_default_joint_pos_[i] + action[i] * action_scale_[i];
    }
    return clamp_vec(q, joint_pos_lower_limit_, joint_pos_upper_limit_);
}

void State_OmniXtreme::reset_tracking_state(bool keep_current_traj)
{
    if (!keep_current_traj)
    {
        trajectory_index_ = 0;
    }
    frame_index_ = trajectories_[trajectory_index_].ref_joint_pos_frames.size() > 1 ? 1 : 0;
    real_obs_history_.clear();
    last_action_.assign(dof_, 0.0f);
    last_base_action_.assign(dof_, 0.0f);
    latest_tau_ff_.assign(dof_, 0.0f);
}

void State_OmniXtreme::handle_gamepad_events()
{
    const auto& joy = FSMState::lowstate->joystick;

    if (toggle_execute_trigger_ && toggle_execute_trigger_(joy))
    {
        const bool start_execute = !execute_motion_.load();
        execute_motion_ = start_execute;
        reset_tracking_state(true);
        calibrate_yaw_alignment();
        spdlog::info("State_OmniXtreme: {} trajectory {} ({}/{})",
                     start_execute ? "start" : "pause",
                     trajectories_[trajectory_index_].name,
                     trajectory_index_ + 1,
                     trajectories_.size());
    }

    if (next_trajectory_trigger_ && next_trajectory_trigger_(joy))
    {
        trajectory_index_ = (trajectory_index_ + 1) % trajectories_.size();
        reset_tracking_state(true);
        calibrate_yaw_alignment();
        spdlog::info("State_OmniXtreme: switch trajectory to {} ({}/{})",
                     trajectories_[trajectory_index_].name, trajectory_index_ + 1, trajectories_.size());
    }

    if (previous_trajectory_trigger_ && previous_trajectory_trigger_(joy))
    {
        trajectory_index_ = (trajectory_index_ + trajectories_.size() - 1) % trajectories_.size();
        reset_tracking_state(true);
        calibrate_yaw_alignment();
        spdlog::info("State_OmniXtreme: switch trajectory to {} ({}/{})",
                     trajectories_[trajectory_index_].name, trajectory_index_ + 1, trajectories_.size());
    }

    if (reset_trajectory_trigger_ && reset_trajectory_trigger_(joy))
    {
        reset_tracking_state(true);
        calibrate_yaw_alignment();
        spdlog::info("State_OmniXtreme: reset trajectory {}", trajectories_[trajectory_index_].name);
    }
}

void State_OmniXtreme::advance_frame()
{
    const std::size_t frame_count = trajectories_[trajectory_index_].ref_joint_pos_frames.size();
    if (frame_count == 0)
    {
        return;
    }

    if (frame_index_ + 1 < frame_count)
    {
        ++frame_index_;
    }
    else if (loop_trajectory_)
    {
        frame_index_ = frame_count > 1 ? 1 : 0;
    }
    else
    {
        frame_index_ = frame_count - 1;
    }
}

std::size_t State_OmniXtreme::paused_frame_index() const
{
    if (trajectories_.empty())
    {
        return 0;
    }
    const std::size_t frame_count = trajectories_[trajectory_index_].ref_joint_pos_frames.size();
    return frame_count > 0 ? 0 : 0;
}

std::vector<float> State_OmniXtreme::to_float_vector(const cnpy::NpyArray& array)
{
    if (array.word_size != sizeof(float))
    {
        throw std::runtime_error("State_OmniXtreme: only float32 npz arrays are supported");
    }

    const float* data_ptr = array.data<float>();
    const std::size_t count = product_of_shape(array.shape);
    return std::vector<float>(data_ptr, data_ptr + count);
}
