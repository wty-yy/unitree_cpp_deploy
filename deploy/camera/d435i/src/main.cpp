#include "Config.h"
#include "D435iCamera.h"
#include "DepthPublisher.h"

#include <unitree/robot/channel/channel_factory.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>

#ifndef D435I_CONFIG_PATH
#error "D435I_CONFIG_PATH must be provided by CMake"
#endif

namespace
{

std::atomic<bool> running{true};

void HandleSignal(int)
{
    running.store(false);
}

}  // namespace

int main()
{
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    try {
        const std::filesystem::path config_path = D435I_CONFIG_PATH;
        const d435i::AppConfig config = d435i::LoadConfig(config_path);
        std::cout << "Configuration: " << config_path << std::endl;

        unitree::robot::ChannelFactory::Instance()->Init(
            config.dds.domain_id, config.dds.interface);

        d435i::D435iCamera camera(config.camera, config.validation);
        camera.Start();

        std::vector<float> warmup_buffer;
        for (int index = 0; index < config.camera.warmup_frames && running.load(); ++index) {
            camera.WaitForFrame(warmup_buffer);
        }
        std::cout << "Camera warmup complete: " << config.camera.warmup_frames
                  << " frames" << std::endl;

        d435i::DepthPublisher publisher(config.dds, config.camera);
        auto& depth = publisher.depth_buffer();
        const int frame_stride = config.camera.capture_fps / config.camera.publish_hz;
        std::uint64_t captured_frames = 0;
        std::uint64_t interval_published_frames = 0;
        auto interval_start = std::chrono::steady_clock::now();

        std::cout << "Publishing " << config.camera.width << "x"
                  << config.camera.height << " 32FC1 depth on " << config.dds.topic
                  << " at " << config.camera.publish_hz << " Hz" << std::endl;

        while (running.load()) {
            const d435i::DepthFrameInfo frame_info = camera.WaitForFrame(depth);
            ++captured_frames;
            if (captured_frames % static_cast<std::uint64_t>(frame_stride) != 0) {
                continue;
            }

            if (!publisher.Publish(frame_info)) {
                std::cerr << "[WARN] Failed to publish depth frame "
                          << publisher.published_frames() - 1 << std::endl;
            }
            ++interval_published_frames;

            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - interval_start).count();
            if (elapsed >= 2.0) {
                std::cout << "Published "
                          << static_cast<double>(interval_published_frames) / elapsed
                          << " Hz, source frame " << frame_info.source_frame_number
                          << std::endl;
                interval_start = now;
                interval_published_frames = 0;
            }
        }

        camera.Stop();
        std::cout << "Stopped after publishing " << publisher.published_frames()
                  << " depth frames" << std::endl;
        return 0;
    } catch (const rs2::error& error) {
        std::cerr << "RealSense error in " << error.get_failed_function() << "("
                  << error.get_failed_args() << "): " << error.what() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "D435i publisher failed: " << error.what() << std::endl;
    }

    return 1;
}
