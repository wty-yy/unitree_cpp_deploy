#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>

// Passive subscribers: this recorder never publishes motor commands.
class G1LoggingTelemetry
{
public:
    using Clock = std::chrono::steady_clock;
    struct Snapshot {
        unitree_hg::msg::dds_::IMUState_ torso;
        unitree_hg::msg::dds_::LowCmd_ lowcmd;
        std::uint64_t torso_sequence = 0;
        std::uint64_t lowcmd_sequence = 0;
        Clock::time_point torso_received;
        Clock::time_point lowcmd_received;
    };

    G1LoggingTelemetry(bool torso_imu, bool lowcmd)
    {
        if (torso_imu) {
            torso_ = std::make_unique<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>>("rt/secondary_imu");
            torso_->InitChannel([this](const void* message) {
                std::lock_guard<std::mutex> lock(mutex_);
                latest_.torso = *static_cast<const unitree_hg::msg::dds_::IMUState_*>(message);
                latest_.torso_received = Clock::now();
                ++latest_.torso_sequence;
            }, 1);
        }
        if (lowcmd) {
            lowcmd_ = std::make_unique<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>>("rt/lowcmd");
            lowcmd_->InitChannel([this](const void* message) {
                std::lock_guard<std::mutex> lock(mutex_);
                latest_.lowcmd = *static_cast<const unitree_hg::msg::dds_::LowCmd_*>(message);
                latest_.lowcmd_received = Clock::now();
                ++latest_.lowcmd_sequence;
            }, 1);
        }
    }

    ~G1LoggingTelemetry()
    {
        if (torso_) torso_->CloseChannel();
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
    std::unique_ptr<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>> torso_;
    std::unique_ptr<unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_>> lowcmd_;
};
