#pragma once

#include "Config.h"
#include "D435iCamera.h"
#include "dds/DepthObservation_.hpp"

#include <unitree/robot/channel/channel_publisher.hpp>

#include <cstdint>
#include <vector>

namespace d435i
{

class DepthPublisher
{
public:
    using Message = unitree_sim::msg::dds_::DepthObservation_;

    DepthPublisher(const DdsConfig& dds_config, const CameraConfig& camera_config);
    ~DepthPublisher();

    DepthPublisher(const DepthPublisher&) = delete;
    DepthPublisher& operator=(const DepthPublisher&) = delete;

    std::vector<float>& depth_buffer() { return message_.data(); }
    bool Publish(const DepthFrameInfo& frame_info);
    std::uint64_t published_frames() const { return published_frames_; }

private:
    unitree::robot::ChannelPublisher<Message> publisher_;
    Message message_;
    std::uint64_t published_frames_ = 0;
};

}  // namespace d435i
