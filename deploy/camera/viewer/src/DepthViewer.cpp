#include "DepthViewer.h"

#include <iostream>
#include <utility>

namespace camera_viewer
{

DepthViewer::DepthViewer(DdsConfig dds_config, StreamConfig stream_config)
    : dds_config_(std::move(dds_config)),
      stream_config_(std::move(stream_config)),
      subscriber_(dds_config_.topic)
{
}

DepthViewer::~DepthViewer()
{
    Stop();
}

void DepthViewer::Start()
{
    if (started_) {
        return;
    }
    subscriber_.InitChannel(
        [this](const void* message) { OnMessage(message); },
        dds_config_.queue_length);
    started_ = true;
}

void DepthViewer::Stop()
{
    if (!started_) {
        return;
    }
    subscriber_.CloseChannel();
    started_ = false;
}

bool DepthViewer::WaitForFrame(DepthFrame& frame, std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(
            lock, timeout, [this]() { return generation_ != consumed_generation_; }))
    {
        return false;
    }
    frame = latest_frame_;
    consumed_generation_ = generation_;
    return true;
}

void DepthViewer::OnMessage(const void* raw_message)
{
    if (raw_message == nullptr) {
        WarnInvalidFrameOnce("received a null DDS sample");
        return;
    }

    const auto& message = *static_cast<const Message*>(raw_message);
    const std::size_t expected_size =
        static_cast<std::size_t>(stream_config_.width) * stream_config_.height;
    if (message.width() != static_cast<std::uint32_t>(stream_config_.width) ||
        message.height() != static_cast<std::uint32_t>(stream_config_.height))
    {
        WarnInvalidFrameOnce("unexpected frame dimensions");
        return;
    }
    if (message.encoding() != stream_config_.encoding) {
        WarnInvalidFrameOnce("unexpected frame encoding " + message.encoding());
        return;
    }
    if (message.data().size() != expected_size) {
        WarnInvalidFrameOnce("unexpected depth payload size");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_frame_.sequence = message.frame_sequence();
        latest_frame_.timestamp_seconds = message.sim_time();
        latest_frame_.data = message.data();
        ++generation_;
        invalid_warning_emitted_ = false;
    }
    condition_.notify_one();
}

void DepthViewer::WarnInvalidFrameOnce(const std::string& reason)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!invalid_warning_emitted_) {
        std::cerr << "[WARN] Ignoring depth frame on " << dds_config_.topic
                  << ": " << reason << std::endl;
        invalid_warning_emitted_ = true;
    }
}

}  // namespace camera_viewer
