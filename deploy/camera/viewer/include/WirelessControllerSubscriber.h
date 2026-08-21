#pragma once

#include "ViewerConfig.h"

#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <atomic>

namespace camera_viewer
{

class WirelessControllerSubscriber
{
public:
    using Message = unitree_go::msg::dds_::WirelessController_;

    explicit WirelessControllerSubscriber(const DdsConfig& config);
    ~WirelessControllerSubscriber();

    WirelessControllerSubscriber(const WirelessControllerSubscriber&) = delete;
    WirelessControllerSubscriber& operator=(const WirelessControllerSubscriber&) = delete;

    void Start();
    void Stop();
    unsigned int ConsumeToggleCount();

private:
    void OnMessage(const void* raw_message);

    unitree::robot::ChannelSubscriber<Message> subscriber_;
    int queue_length_ = 0;
    std::atomic<unsigned int> pending_toggles_{0};
    bool last_chord_pressed_ = false;
    bool started_ = false;
};

}  // namespace camera_viewer
