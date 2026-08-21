#pragma once

#include "ViewerConfig.h"

#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>
#include <optional>

namespace camera_viewer
{

class RecordSwitchSubscriber
{
public:
    using Message = std_msgs::msg::dds_::String_;

    explicit RecordSwitchSubscriber(const DdsConfig& config);
    ~RecordSwitchSubscriber();

    RecordSwitchSubscriber(const RecordSwitchSubscriber&) = delete;
    RecordSwitchSubscriber& operator=(const RecordSwitchSubscriber&) = delete;

    void Start();
    void Stop();
    std::optional<bool> ConsumeCommand();

private:
    void OnMessage(const void* raw_message);

    unitree::robot::ChannelSubscriber<Message> subscriber_;
    int queue_length_ = 0;
    std::atomic<int> requested_state_{-1};
    std::atomic<bool> invalid_warning_emitted_{false};
    bool started_ = false;
};

}  // namespace camera_viewer
