#pragma once

#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <vector>

namespace utils {

// Two scalar HeightMap_ messages share a wall-clock stamp. DDS arrays use
// y-major rows; inference receives [relative_elevation, missing_mask, x, y].
// Heights are in metres; the binary mask is 0 for known and 1 for missing.
class PolicyHeightMap {
    using Message = unitree_go::msg::dds_::HeightMap_;
public:
    explicit PolicyHeightMap(bool forward_elevation = false)
        : width_(forward_elevation ? 9 : 17), cells_(width_ * 11) {}

    void start(const std::string& topic = "rt/g1/policy_height_map", double timeout = 0.5)
    {
        if (!std::isfinite(timeout) || timeout <= 0) throw std::invalid_argument("Height-map timeout must be positive.");
        timeout_ = timeout;
        elevation_sub_ = std::make_unique<unitree::robot::ChannelSubscriber<Message>>(topic);
        missing_sub_ = std::make_unique<unitree::robot::ChannelSubscriber<Message>>(topic + "_missing_mask");
        elevation_sub_->InitChannel([this](const void* data) { receive(*static_cast<const Message*>(data), false); }, 1);
        missing_sub_->InitChannel([this](const void* data) { receive(*static_cast<const Message*>(data), true); }, 1);
    }

    ~PolicyHeightMap()
    {
        if (elevation_sub_) elevation_sub_->CloseChannel();
        if (missing_sub_) missing_sub_->CloseChannel();
    }

    bool sample(std::vector<float>& image)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        const double age = now - stamp_;
        if (image_.empty() || !std::any_of(image_.begin() + cells_, image_.end(), [](float value) { return value == 0.0f; }) ||
            age < -0.1 || age > timeout_ ||
            std::chrono::duration<double>(std::chrono::steady_clock::now() - received_).count() > timeout_) {
            image.assign(2 * cells_, 0.0f);
            std::fill(image.begin() + cells_, image.end(), 1.0f);
            return false;
        }
        image = image_;
        return true;
    }

private:
    void receive(const Message& message, bool missing)
    {
        const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (!std::isfinite(message.stamp()) || now - message.stamp() < -0.1 || now - message.stamp() > timeout_) return;
        const int source_width = message.width();
        const bool full_map = source_width == 17 && std::abs(message.origin()[0] + 0.8f) <= 1e-5f;
        const bool forward_map = source_width == 9 && std::abs(message.origin()[0]) <= 1e-5f;
        if (message.frame_id() != "base_yaw_relative" || !(full_map || (width_ == 9 && forward_map)) || message.height() != 11 ||
            !std::isfinite(message.resolution()) || std::abs(message.resolution() - 0.1f) > 1e-5f ||
            !std::isfinite(message.origin()[1]) ||
            std::abs(message.origin()[1] + 0.5f) > 1e-5f ||
            message.data().size() != static_cast<size_t>(source_width * 11) ||
            !std::all_of(message.data().begin(), message.data().end(), [missing](float value) {
                return std::isfinite(value) && (missing ? (value == 0.0f || value == 1.0f) : std::abs(value) <= 2.0f);
            })) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto& pending = missing ? missing_ : elevation_;
        if (message.stamp() <= stamp_ || message.stamp() < pending.stamp()) return;
        pending = message;
        if (elevation_.stamp() != missing_.stamp() || elevation_.width() != missing_.width()) return;
        image_.resize(2 * cells_);
        const int offset = source_width - width_;
        for (int x = 0; x < width_; ++x) for (int y = 0; y < 11; ++y) {
            const int source = y * source_width + x + offset;
            const float mask = missing_.data()[source];
            image_[x * 11 + y] = mask == 0.0f ? elevation_.data()[source] : 0.0f;
            image_[cells_ + x * 11 + y] = mask;
        }
        stamp_ = message.stamp();
        received_ = std::chrono::steady_clock::now();
    }

    std::unique_ptr<unitree::robot::ChannelSubscriber<Message>> elevation_sub_, missing_sub_;
    std::mutex mutex_;
    Message elevation_, missing_;
    std::vector<float> image_;
    double stamp_ = 0, timeout_ = 0.5;
    std::chrono::steady_clock::time_point received_;
    const int width_, cells_;
};
}
