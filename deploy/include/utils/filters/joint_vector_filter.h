#pragma once

#include "joint_kalman_filter.h"
#include "joint_lpf_filter.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace joint_filter
{

enum class JointFilterType
{
    None,
    Lpf,
    Kalman
};

struct JointFilterConfig
{
    bool enabled{false};
    JointFilterType type{JointFilterType::None};
    bool has_joint_indices{false};
    std::vector<int> joint_indices;
    JointLpfFilterConfig lpf;
    JointKalmanFilterConfig kalman;
};

inline std::vector<int> default_waist_joint_indices(std::size_t dof)
{
    if (dof > 14)
    {
        return {12, 13, 14};
    }
    return {};
}

inline std::string to_string(JointFilterType type)
{
    switch (type)
    {
    case JointFilterType::Lpf:
        return "lpf";
    case JointFilterType::Kalman:
        return "kalman";
    default:
        return "none";
    }
}

inline JointFilterType joint_filter_type_from_string(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value.empty() || value == "none" || value == "disabled")
    {
        return JointFilterType::None;
    }
    if (value == "lpf" || value == "lowpass" || value == "low_pass")
    {
        return JointFilterType::Lpf;
    }
    if (value == "kalman")
    {
        return JointFilterType::Kalman;
    }
    throw std::runtime_error("JointVectorFilter: unsupported filter type '" + value + "'");
}

inline JointFilterConfig joint_filter_config_from_yaml(const YAML::Node& node)
{
    JointFilterConfig config;
    if (!node || node.IsNull())
    {
        return config;
    }

    config.enabled = node["enabled"] ? node["enabled"].as<bool>() : true;
    if (node["type"])
    {
        config.type = joint_filter_type_from_string(node["type"].as<std::string>());
    }
    if (node["joint_indices"])
    {
        config.has_joint_indices = true;
        config.joint_indices = node["joint_indices"].as<std::vector<int>>();
    }
    if (node["alpha"])
    {
        config.lpf.alpha = node["alpha"].as<float>();
    }
    if (node["lpf"] && node["lpf"]["alpha"])
    {
        config.lpf.alpha = node["lpf"]["alpha"].as<float>();
    }
    if (node["process_noise"])
    {
        config.kalman.process_noise = node["process_noise"].as<float>();
    }
    if (node["measurement_noise"])
    {
        config.kalman.measurement_noise = node["measurement_noise"].as<float>();
    }
    if (node["initial_error_cov"])
    {
        config.kalman.initial_error_cov = node["initial_error_cov"].as<float>();
    }
    if (node["kalman"])
    {
        const auto kalman = node["kalman"];
        if (kalman["process_noise"])
        {
            config.kalman.process_noise = kalman["process_noise"].as<float>();
        }
        if (kalman["measurement_noise"])
        {
            config.kalman.measurement_noise = kalman["measurement_noise"].as<float>();
        }
        if (kalman["initial_error_cov"])
        {
            config.kalman.initial_error_cov = kalman["initial_error_cov"].as<float>();
        }
    }
    if (config.type == JointFilterType::None && config.enabled)
    {
        if (node["kalman"] || node["process_noise"] || node["measurement_noise"] || node["initial_error_cov"])
        {
            config.type = JointFilterType::Kalman;
        }
        else if (node["lpf"] || node["alpha"])
        {
            config.type = JointFilterType::Lpf;
        }
    }

    return config;
}

class JointVectorFilter
{
public:
    void configure(
        const JointFilterConfig& config,
        std::size_t dof,
        std::string name,
        std::vector<int> default_joint_indices = {})
    {
        clear();
        dof_ = dof;
        name_ = std::move(name);
        config_ = config;

        if (!config_.enabled || config_.type == JointFilterType::None)
        {
            spdlog::info("{}: disabled", name_);
            return;
        }

        std::vector<int> joint_indices = config_.has_joint_indices
            ? config_.joint_indices
            : (default_joint_indices.empty() ? all_joint_indices(dof_) : std::move(default_joint_indices));
        joint_indices = sanitize_joint_indices(joint_indices);
        if (joint_indices.empty())
        {
            config_.enabled = false;
            config_.type = JointFilterType::None;
            spdlog::warn("{}: disabled because no valid joint indices remain", name_);
            return;
        }

        joint_indices_ = joint_indices;
        if (config_.type == JointFilterType::Lpf)
        {
            lpf_ = std::make_unique<JointLpfFilter>(dof_, joint_indices_, config_.lpf);
            spdlog::info(
                "{}: enabled type=lpf joints={} alpha={}",
                name_,
                join_indices(joint_indices_),
                lpf_->alpha());
            return;
        }

        if (config_.type == JointFilterType::Kalman)
        {
            kalman_ = std::make_unique<JointKalmanFilter>(dof_, joint_indices_, config_.kalman);
            spdlog::info(
                "{}: enabled type=kalman joints={} q={} r={} p0={}",
                name_,
                join_indices(joint_indices_),
                kalman_->process_noise(),
                kalman_->measurement_noise(),
                kalman_->initial_error_cov());
            return;
        }

        throw std::runtime_error(name_ + ": unsupported filter configuration");
    }

    void configure(
        const YAML::Node& node,
        std::size_t dof,
        std::string name,
        std::vector<int> default_joint_indices = {})
    {
        configure(joint_filter_config_from_yaml(node), dof, std::move(name), std::move(default_joint_indices));
    }

    void reset(const std::vector<float>& values)
    {
        validate_size(values);
        if (lpf_)
        {
            lpf_->reset(values);
        }
        if (kalman_)
        {
            kalman_->reset(values);
        }
    }

    void apply(std::vector<float>& values)
    {
        validate_size(values);
        if (lpf_)
        {
            lpf_->apply(values);
        }
        if (kalman_)
        {
            kalman_->apply(values);
        }
    }

    bool enabled() const
    {
        return lpf_ || kalman_;
    }

private:
    static std::vector<int> all_joint_indices(std::size_t dof)
    {
        std::vector<int> out;
        out.reserve(dof);
        for (std::size_t i = 0; i < dof; ++i)
        {
            out.push_back(static_cast<int>(i));
        }
        return out;
    }

    static std::string join_indices(const std::vector<int>& indices)
    {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            if (i != 0)
            {
                oss << ", ";
            }
            oss << indices[i];
        }
        oss << "]";
        return oss.str();
    }

    std::vector<int> sanitize_joint_indices(const std::vector<int>& raw_indices) const
    {
        std::vector<int> sanitized;
        sanitized.reserve(raw_indices.size());
        for (int idx : raw_indices)
        {
            if (idx < 0 || static_cast<std::size_t>(idx) >= dof_)
            {
                spdlog::warn("{}: joint index {} out of range [0, {})", name_, idx, dof_);
                continue;
            }
            if (std::find(sanitized.begin(), sanitized.end(), idx) == sanitized.end())
            {
                sanitized.push_back(idx);
            }
        }
        return sanitized;
    }

    void clear()
    {
        dof_ = 0;
        name_.clear();
        config_ = JointFilterConfig{};
        joint_indices_.clear();
        lpf_.reset();
        kalman_.reset();
    }

    void validate_size(const std::vector<float>& values) const
    {
        if (enabled() && values.size() != dof_)
        {
            throw std::runtime_error(name_ + ": input size mismatch");
        }
    }

    std::size_t dof_{0};
    std::string name_;
    JointFilterConfig config_;
    std::vector<int> joint_indices_;
    std::unique_ptr<JointLpfFilter> lpf_;
    std::unique_ptr<JointKalmanFilter> kalman_;
};

class JointVectorFilterBank
{
public:
    void configure(
        const YAML::Node& node,
        std::size_t dof,
        const std::string& name,
        std::vector<int> default_joint_indices = {})
    {
        clear();
        if (!node || node.IsNull())
        {
            spdlog::info("{}: disabled", name);
            return;
        }

        if (node.IsSequence())
        {
            std::size_t index = 0;
            for (const auto& item : node)
            {
                JointVectorFilter filter;
                filter.configure(
                    item,
                    dof,
                    name + "[" + std::to_string(index) + "]",
                    default_joint_indices);
                filters_.push_back(std::move(filter));
                ++index;
            }
            return;
        }

        JointVectorFilter filter;
        filter.configure(node, dof, name, std::move(default_joint_indices));
        filters_.push_back(std::move(filter));
    }

    void configure(
        const std::vector<JointFilterConfig>& configs,
        std::size_t dof,
        const std::string& name,
        std::vector<int> default_joint_indices = {})
    {
        clear();
        for (std::size_t i = 0; i < configs.size(); ++i)
        {
            JointVectorFilter filter;
            filter.configure(
                configs[i],
                dof,
                name + "[" + std::to_string(i) + "]",
                default_joint_indices);
            filters_.push_back(std::move(filter));
        }
        if (filters_.empty())
        {
            spdlog::info("{}: disabled", name);
        }
    }

    void reset(const std::vector<float>& values)
    {
        for (auto& filter : filters_)
        {
            filter.reset(values);
        }
    }

    void apply(std::vector<float>& values)
    {
        for (auto& filter : filters_)
        {
            filter.apply(values);
        }
    }

    bool enabled() const
    {
        for (const auto& filter : filters_)
        {
            if (filter.enabled())
            {
                return true;
            }
        }
        return false;
    }

private:
    void clear()
    {
        filters_.clear();
    }

    std::vector<JointVectorFilter> filters_;
};

} // namespace joint_filter
