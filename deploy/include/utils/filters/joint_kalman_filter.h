#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace joint_filter
{

struct JointKalmanFilterConfig
{
    float process_noise{1.0e-4f};
    float measurement_noise{2.0e-3f};
    float initial_error_cov{1.0e-2f};
};

class JointKalmanFilter
{
public:
    JointKalmanFilter(
        std::size_t dof,
        std::vector<int> joint_indices,
        JointKalmanFilterConfig config)
    : dof_(dof),
      joint_indices_(std::move(joint_indices)),
      config_(config),
      state_(dof, 0.0f),
      error_cov_(dof, config.initial_error_cov),
      initialized_(dof, 0)
    {
        config_.process_noise = std::max(config_.process_noise, 1.0e-8f);
        config_.measurement_noise = std::max(config_.measurement_noise, 1.0e-8f);
        config_.initial_error_cov = std::max(config_.initial_error_cov, 1.0e-8f);
        std::fill(error_cov_.begin(), error_cov_.end(), config_.initial_error_cov);
    }

    void reset(const std::vector<float>& values)
    {
        validate_size(values);
        for (int idx : joint_indices_)
        {
            state_[idx] = values[idx];
            error_cov_[idx] = config_.initial_error_cov;
            initialized_[idx] = 1;
        }
    }

    void apply(std::vector<float>& values)
    {
        validate_size(values);
        for (int idx : joint_indices_)
        {
            if (!initialized_[idx])
            {
                state_[idx] = values[idx];
                error_cov_[idx] = config_.initial_error_cov;
                initialized_[idx] = 1;
                continue;
            }

            const float measurement = values[idx];
            float prediction = state_[idx];
            float prediction_cov = error_cov_[idx] + config_.process_noise;
            const float gain = prediction_cov / (prediction_cov + config_.measurement_noise);

            prediction = prediction + gain * (measurement - prediction);
            prediction_cov = (1.0f - gain) * prediction_cov;

            state_[idx] = prediction;
            error_cov_[idx] = std::max(prediction_cov, 1.0e-10f);
            values[idx] = prediction;
        }
    }

    float process_noise() const
    {
        return config_.process_noise;
    }

    float measurement_noise() const
    {
        return config_.measurement_noise;
    }

    float initial_error_cov() const
    {
        return config_.initial_error_cov;
    }

private:
    void validate_size(const std::vector<float>& values) const
    {
        if (values.size() != dof_)
        {
            throw std::runtime_error("JointKalmanFilter: input size mismatch");
        }
    }

    std::size_t dof_;
    std::vector<int> joint_indices_;
    JointKalmanFilterConfig config_;
    std::vector<float> state_;
    std::vector<float> error_cov_;
    std::vector<std::uint8_t> initialized_;
};

} // namespace joint_filter
