#include "DepthPublisher.h"

#include <stdexcept>

namespace d435i
{

DepthPublisher::DepthPublisher(
    const DdsConfig& dds_config, const CameraConfig& camera_config)
    : publisher_(dds_config.topic)
{
    publisher_.InitChannel();
    message_.frame_id(camera_config.frame_id);
    message_.width(static_cast<std::uint32_t>(camera_config.width));
    message_.height(static_cast<std::uint32_t>(camera_config.height));
    message_.encoding("32FC1");
    message_.near_clip(camera_config.near_clip);
    message_.far_clip(camera_config.far_clip);
    message_.data().resize(
        static_cast<std::size_t>(camera_config.width) * camera_config.height);
}

DepthPublisher::~DepthPublisher()
{
    publisher_.CloseChannel();
}

bool DepthPublisher::Publish(const DepthFrameInfo& frame_info)
{
    message_.frame_sequence(published_frames_);
    message_.sim_time(frame_info.timestamp_seconds);
    const bool success = publisher_.Write(message_);
    ++published_frames_;
    return success;
}

}  // namespace d435i
