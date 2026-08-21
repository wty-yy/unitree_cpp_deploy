#pragma once

#include "ViewerConfig.h"
#include "dds/DepthObservation_.hpp"

#include <unitree/robot/channel/channel_subscriber.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace camera_viewer
{

struct DepthFrame
{
    std::uint64_t sequence = 0;
    double timestamp_seconds = 0.0;
    std::vector<float> data;
};

class DepthViewer
{
public:
    using Message = unitree_sim::msg::dds_::DepthObservation_;

    DepthViewer(DdsConfig dds_config, StreamConfig stream_config);
    ~DepthViewer();

    DepthViewer(const DepthViewer&) = delete;
    DepthViewer& operator=(const DepthViewer&) = delete;

    void Start();
    void Stop();
    bool WaitForFrame(DepthFrame& frame, std::chrono::milliseconds timeout);

private:
    void OnMessage(const void* raw_message);
    void WarnInvalidFrameOnce(const std::string& reason);

    DdsConfig dds_config_;
    StreamConfig stream_config_;
    unitree::robot::ChannelSubscriber<Message> subscriber_;
    std::mutex mutex_;
    std::condition_variable condition_;
    DepthFrame latest_frame_;
    std::uint64_t generation_ = 0;
    std::uint64_t consumed_generation_ = 0;
    bool invalid_warning_emitted_ = false;
    bool started_ = false;
};

}  // namespace camera_viewer
