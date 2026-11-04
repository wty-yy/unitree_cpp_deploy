#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>

// Passive subscribers: this recorder never publishes motor commands.
class Go2LoggingTelemetry
{
public:
    using Clock = std::chrono::steady_clock;
    struct Snapshot {
        unitree_go::msg::dds_::LowCmd_ lowcmd;
        std::uint64_t lowcmd_sequence = 0;
        Clock::time_point lowcmd_received;
    };

    Go2LoggingTelemetry()
    {
        lowcmd_ = std::make_unique<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowCmd_>>("rt/lowcmd");
        lowcmd_->InitChannel([this](const void* message) {
            std::lock_guard<std::mutex> lock(mutex_);
            latest_.lowcmd = *static_cast<const unitree_go::msg::dds_::LowCmd_*>(message);
            latest_.lowcmd_received = Clock::now();
            ++latest_.lowcmd_sequence;
        }, 1);
    }

    ~Go2LoggingTelemetry()
    {
        if (lowcmd_) lowcmd_->CloseChannel();
    }

    Snapshot snapshot()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_;
    }

private:
    std::mutex mutex_;
    Snapshot latest_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowCmd_>> lowcmd_;
};
