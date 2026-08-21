// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "dds/DepthObservation_.hpp"
#include "depth/D435iDepthProcessor.h"
#include "isaaclab/envs/manager_based_rl_env.h"

#include <unitree/robot/channel/channel_subscriber.hpp>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

namespace isaaclab
{
namespace mdp
{
namespace detail
{

inline deploy::D435iDepthProcessor::Config parse_depth_processor_config(
    const YAML::Node& params)
{
    deploy::D435iDepthProcessor::Config config;
    if (params["raw_width"]) {
        config.raw_width = params["raw_width"].as<int>();
    }
    if (params["raw_height"]) {
        config.raw_height = params["raw_height"].as<int>();
    }
    if (params["output_width"]) {
        config.output_width = params["output_width"].as<int>();
    }
    if (params["output_height"]) {
        config.output_height = params["output_height"].as<int>();
    }
    if (params["min_depth"]) {
        config.min_depth = params["min_depth"].as<float>();
    }
    if (params["max_depth"]) {
        config.max_depth = params["max_depth"].as<float>();
    }
    if (params["gaussian_kernel_size"]) {
        config.gaussian_kernel_size = params["gaussian_kernel_size"].as<int>();
    }
    if (params["gaussian_sigma"]) {
        config.gaussian_sigma = params["gaussian_sigma"].as<double>();
    }
    return config;
}

class DepthObservationSource
{
public:
    using Message = unitree_sim::msg::dds_::DepthObservation_;

    struct Sample
    {
        std::vector<float> data;
        bool valid = false;
    };

    explicit DepthObservationSource(const YAML::Node& params)
        : topic_(params["topic"]
              ? params["topic"].as<std::string>()
              : "rt/front_depth_observation"),
          expected_encoding_(params["encoding"]
              ? params["encoding"].as<std::string>()
              : "32FC1"),
          processor_(parse_depth_processor_config(params)),
          stale_timeout_(params["stale_timeout"]
              ? params["stale_timeout"].as<double>()
              : 0.5),
          fallback_value_(params["fallback_value"]
              ? params["fallback_value"].as<float>()
              : 1.0F),
          saturation_value_(params["saturation_value"]
              ? params["saturation_value"].as<float>()
              : 1.0F),
          max_saturated_fraction_(params["max_saturated_fraction"]
              ? params["max_saturated_fraction"].as<float>()
              : 1.0F)
    {
        if (topic_.empty()) {
            throw std::invalid_argument("Depth observation topic must not be empty");
        }
        if (expected_encoding_.empty()) {
            throw std::invalid_argument("Depth observation encoding must not be empty");
        }
        if (!std::isfinite(stale_timeout_.count()) || stale_timeout_.count() < 0.0) {
            throw std::invalid_argument("Depth stale_timeout must be finite and non-negative");
        }
        if (!std::isfinite(fallback_value_) || fallback_value_ < 0.0F ||
            fallback_value_ > 1.0F)
        {
            throw std::invalid_argument("Depth fallback_value must be within [0, 1]");
        }
        if (!std::isfinite(saturation_value_) || saturation_value_ < 0.0F ||
            saturation_value_ > 1.0F)
        {
            throw std::invalid_argument("Depth saturation_value must be within [0, 1]");
        }
        if (!std::isfinite(max_saturated_fraction_) ||
            max_saturated_fraction_ < 0.0F || max_saturated_fraction_ > 1.0F)
        {
            throw std::invalid_argument(
                "Depth max_saturated_fraction must be within [0, 1]");
        }

        const auto& config = processor_.config();
        fallback_.assign(
            static_cast<std::size_t>(config.output_width) * config.output_height,
            fallback_value_);

        const auto queue_length = params["queue_length"]
            ? params["queue_length"].as<std::int64_t>()
            : 0;
        if (queue_length < 0) {
            throw std::invalid_argument("Depth queue_length must be non-negative");
        }

        subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<Message>>(topic_);
        subscriber_->InitChannel(
            std::bind(&DepthObservationSource::on_message, this, std::placeholders::_1),
            queue_length);
        spdlog::info(
            "Depth observation subscribed to {} ({}x{} {} -> {}x{})",
            topic_, config.raw_width, config.raw_height, expected_encoding_,
            config.output_width, config.output_height);
    }

    ~DepthObservationSource()
    {
        if (subscriber_) {
            subscriber_->CloseChannel();
        }
    }

    Sample observation()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();
        const bool stale = stale_timeout_.count() > 0.0 && has_frame_
            && now - last_frame_time_ > stale_timeout_;
        if (has_frame_ && latest_frame_valid_ && !stale) {
            unavailable_warning_emitted_ = false;
            return {latest_, true};
        }

        if (!unavailable_warning_emitted_) {
            if (has_frame_ && !latest_frame_valid_) {
                spdlog::warn(
                    "Latest depth observation on {} failed quality checks; "
                    "using fallback value {:.3f}",
                    topic_, fallback_value_);
            } else if (stale) {
                spdlog::warn(
                    "Depth observation on {} is stale; using fallback value {:.3f}",
                    topic_, fallback_value_);
            } else {
                spdlog::warn(
                    "Waiting for first depth observation on {}; using fallback value {:.3f}",
                    topic_, fallback_value_);
            }
            unavailable_warning_emitted_ = true;
        }
        return {fallback_, false};
    }

private:
    void on_message(const void* raw_message)
    {
        if (raw_message == nullptr) {
            mark_latest_frame_invalid();
            warn_invalid_frame_once("received a null message");
            return;
        }

        const auto& message = *static_cast<const Message*>(raw_message);
        if (message.encoding() != expected_encoding_) {
            mark_latest_frame_invalid();
            warn_invalid_frame_once(
                "expected encoding " + expected_encoding_ + ", received " +
                message.encoding());
            return;
        }

        try {
            auto processed = processor_.Process(
                message.data(), message.width(), message.height());

            const auto saturated_count = std::count_if(
                processed.begin(), processed.end(),
                [this](float value) { return value >= saturation_value_; });
            const float saturated_fraction = static_cast<float>(saturated_count) /
                static_cast<float>(processed.size());
            if (saturated_fraction > max_saturated_fraction_) {
                mark_latest_frame_invalid();
                warn_invalid_frame_once(
                    "saturated pixel fraction " + std::to_string(saturated_fraction) +
                    " exceeds configured maximum " +
                    std::to_string(max_saturated_fraction_));
                return;
            }

            bool collect_first_frame_stats = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                collect_first_frame_stats = !has_frame_;
            }

            float raw_min = 0.0F;
            float raw_max = 0.0F;
            float processed_min = 0.0F;
            float processed_max = 0.0F;
            double processed_mean = 0.0;
            if (collect_first_frame_stats) {
                const auto raw_range = std::minmax_element(
                    message.data().begin(), message.data().end());
                raw_min = *raw_range.first;
                raw_max = *raw_range.second;
                const auto processed_range = std::minmax_element(
                    processed.begin(), processed.end());
                processed_min = *processed_range.first;
                processed_max = *processed_range.second;
                processed_mean = std::accumulate(
                    processed.begin(), processed.end(), 0.0) /
                    static_cast<double>(processed.size());
            }

            bool is_first_frame = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                is_first_frame = !has_frame_;
                latest_ = std::move(processed);
                last_frame_time_ = std::chrono::steady_clock::now();
                has_frame_ = true;
                latest_frame_valid_ = true;
                unavailable_warning_emitted_ = false;
                invalid_frame_warning_emitted_ = false;
            }
            if (is_first_frame) {
                spdlog::info(
                    "Receiving depth observations on {} (first sequence {}, "
                    "clip [{:.4f}, {:.3f}] m, raw [{:.3f}, {:.3f}] m, "
                    "policy [{:.3f}, {:.3f}], mean {:.3f}, saturated {:.1f}%)",
                    topic_, message.frame_sequence(), message.near_clip(),
                    message.far_clip(), raw_min, raw_max, processed_min,
                    processed_max, processed_mean, saturated_fraction * 100.0F);
            }
        } catch (const std::exception& error) {
            mark_latest_frame_invalid();
            warn_invalid_frame_once(error.what());
        }
    }

    void mark_latest_frame_invalid()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_frame_valid_ = false;
        unavailable_warning_emitted_ = false;
    }

    void warn_invalid_frame_once(const std::string& reason)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!invalid_frame_warning_emitted_) {
            spdlog::warn("Ignoring depth observation on {}: {}", topic_, reason);
            invalid_frame_warning_emitted_ = true;
        }
    }

    std::string topic_;
    std::string expected_encoding_;
    deploy::D435iDepthProcessor processor_;
    std::chrono::duration<double> stale_timeout_;
    float fallback_value_;
    float saturation_value_;
    float max_saturated_fraction_;
    std::vector<float> fallback_;

    std::shared_ptr<unitree::robot::ChannelSubscriber<Message>> subscriber_;
    std::mutex mutex_;
    std::vector<float> latest_;
    std::chrono::steady_clock::time_point last_frame_time_;
    bool has_frame_ = false;
    bool latest_frame_valid_ = false;
    bool unavailable_warning_emitted_ = false;
    bool invalid_frame_warning_emitted_ = false;
};

}  // namespace detail

REGISTER_OBSERVATION(depth_image)
{
    const auto resource_key = std::string("depth_image:") + YAML::Dump(params);
    auto source = env->observation_resource<detail::DepthObservationSource>(
        resource_key,
        [&params]() {
            return std::make_shared<detail::DepthObservationSource>(params);
        });
    auto sample = source->observation();
    if (!sample.valid) {
        env->invalidate_observations();
    }
    return std::move(sample.data);
}

}  // namespace mdp
}  // namespace isaaclab
