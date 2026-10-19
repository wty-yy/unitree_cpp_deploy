#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace joint_filter
{

struct JointLpfFilterConfig
{
    float alpha{1.0f};
};

class JointLpfFilter
{
public:
    JointLpfFilter(
        std::size_t dof,
        std::vector<int> joint_indices,
        JointLpfFilterConfig config)
    : dof_(dof),
      joint_indices_(std::move(joint_indices)),
      config_(config),
      state_(dof, 0.0f),
      initialized_(dof, 0)
    {
        config_.alpha = std::clamp(config_.alpha, 0.0f, 1.0f);
    }

    void reset(const std::vector<float>& values)
    {
        validate_size(values);
        for (int idx : joint_indices_)
        {
            state_[idx] = values[idx];
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
                initialized_[idx] = 1;
                continue;
            }

            state_[idx] = (1.0f - config_.alpha) * state_[idx] + config_.alpha * values[idx];
            values[idx] = state_[idx];
        }
    }

    float alpha() const
    {
        return config_.alpha;
    }

private:
    void validate_size(const std::vector<float>& values) const
    {
        if (values.size() != dof_)
        {
            throw std::runtime_error("JointLpfFilter: input size mismatch");
        }
    }

    std::size_t dof_;
    std::vector<int> joint_indices_;
    JointLpfFilterConfig config_;
    std::vector<float> state_;
    std::vector<std::uint8_t> initialized_;
};

} // namespace joint_filter
