#include "DepthViewer.h"
#include "FfmpegRecorder.h"
#include "RecordSwitchSubscriber.h"
#include "ViewerConfig.h"
#include "WirelessControllerSubscriber.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <termios.h>
#include <unistd.h>
#include <vector>

#ifndef DEPLOY_CAMERA_ROOT_PATH
#define DEPLOY_CAMERA_ROOT_PATH "."
#endif
#ifndef VIEWER_CONFIG_PATH
#error "VIEWER_CONFIG_PATH must be provided by CMake"
#endif

namespace
{

constexpr char kWindowName[] = "DDS depth viewer (R: record, Q/Esc: quit)";
std::atomic<bool> running{true};

class TerminalInput
{
public:
    TerminalInput()
    {
        if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &original_termios_) != 0) {
            return;
        }
        termios raw = original_termios_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        original_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (original_flags_ < 0 || tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            return;
        }
        if (fcntl(STDIN_FILENO, F_SETFL, original_flags_ | O_NONBLOCK) != 0) {
            tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
            return;
        }
        active_ = true;
    }

    ~TerminalInput()
    {
        if (active_) {
            tcsetattr(STDIN_FILENO, TCSANOW, &original_termios_);
            fcntl(STDIN_FILENO, F_SETFL, original_flags_);
        }
    }

    int ReadKey() const
    {
        if (!active_) {
            return -1;
        }
        unsigned char key = 0;
        return read(STDIN_FILENO, &key, 1) == 1 ? key : -1;
    }

private:
    termios original_termios_{};
    int original_flags_ = 0;
    bool active_ = false;
};

void HandleSignal(int)
{
    running.store(false);
}

std::filesystem::path NewVideoPath(const std::filesystem::path& output_directory)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&time, &local_time);
    std::ostringstream name;
    name << "depth_" << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S") << ".mkv";
    return output_directory / name.str();
}

void ConvertToGrayscale(
    const std::vector<float>& metric_depth,
    int width,
    int height,
    float max_depth,
    cv::Mat& grayscale)
{
    const cv::Mat source(
        height, width, CV_32FC1, const_cast<float*>(metric_depth.data()));
    cv::Mat clipped = source.clone();
    cv::patchNaNs(clipped, max_depth);
    cv::max(clipped, 0.0, clipped);
    cv::min(clipped, max_depth, clipped);
    clipped.convertTo(grayscale, CV_8UC1, 255.0 / max_depth);
}

}  // namespace

int main()
{
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    std::signal(SIGPIPE, SIG_IGN);

    try {
        const std::filesystem::path config_path = VIEWER_CONFIG_PATH;
        const camera_viewer::ViewerConfig config =
            camera_viewer::LoadViewerConfig(config_path);
        std::cout << "Configuration: " << config_path << std::endl;

        unitree::robot::ChannelFactory::Instance()->Init(
            config.dds.domain_id, config.dds.interface);
        camera_viewer::DepthViewer viewer(config.dds, config.stream);
        viewer.Start();
        camera_viewer::RecordSwitchSubscriber record_switch(config.dds);
        record_switch.Start();
        camera_viewer::WirelessControllerSubscriber wireless_controller(config.dds);
        wireless_controller.Start();
        std::cout << "Subscribed to " << config.dds.topic << " ("
                  << config.stream.width << "x" << config.stream.height << " "
                  << config.stream.encoding << ")\n"
                  << "Record control topic: " << config.dds.control_topic << "\n"
                  << "Wireless controller topic: "
                  << config.dds.wireless_controller_topic << "\n"
                  << "Controls: terminal R=record, Q=quit; window R=record, Esc=quit"
                  << std::endl;

        std::filesystem::path output_directory = config.recording.output_dir;
        if (output_directory.is_relative()) {
            output_directory =
                std::filesystem::path(DEPLOY_CAMERA_ROOT_PATH) / output_directory;
        }
        std::filesystem::create_directories(output_directory);

        camera_viewer::FfmpegRecorder recorder(config.recording);
        TerminalInput terminal;
        bool recording_requested = false;
        bool window_created = false;
        const auto set_recording = [&](bool enabled) {
            if (recording_requested == enabled) {
                return;
            }
            recording_requested = enabled;
            std::cout << "Recording " << (enabled ? "enabled" : "disabled")
                      << std::endl;
            if (!enabled && recorder.recording()) {
                recorder.Stop();
            }
        };
        const auto toggle_recording = [&]() {
            set_recording(!recording_requested);
        };

        camera_viewer::DepthFrame frame;
        cv::Mat grayscale;
        auto last_frame_time = std::chrono::steady_clock::now();
        auto last_stale_warning = last_frame_time;
        std::uint64_t previous_sequence = 0;
        bool have_sequence = false;

        while (running.load()) {
            const int terminal_key = terminal.ReadKey();
            if (terminal_key == 'q' || terminal_key == 'Q') {
                running.store(false);
                break;
            }
            if (terminal_key == 'r' || terminal_key == 'R') {
                toggle_recording();
            }
            if (const auto requested_state = record_switch.ConsumeCommand()) {
                set_recording(*requested_state);
            }
            const unsigned int controller_toggles =
                wireless_controller.ConsumeToggleCount();
            for (unsigned int index = 0; index < controller_toggles; ++index) {
                toggle_recording();
            }

            if (!viewer.WaitForFrame(frame, std::chrono::milliseconds(100))) {
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - last_frame_time).count() >
                        config.stream.stale_timeout &&
                    std::chrono::duration<double>(now - last_stale_warning).count() >= 1.0)
                {
                    std::cerr << "[WARN] Waiting for depth frames on "
                              << config.dds.topic << std::endl;
                    last_stale_warning = now;
                }
                continue;
            }
            last_frame_time = std::chrono::steady_clock::now();

            if (have_sequence && frame.sequence > previous_sequence + 1) {
                std::cerr << "[WARN] Dropped " << frame.sequence - previous_sequence - 1
                          << " depth frames" << std::endl;
            }
            previous_sequence = frame.sequence;
            have_sequence = true;

            ConvertToGrayscale(
                frame.data,
                config.stream.width,
                config.stream.height,
                config.preview.max_depth,
                grayscale);

            if (recording_requested && !recorder.recording()) {
                recorder.Start(
                    NewVideoPath(output_directory),
                    config.stream.width,
                    config.stream.height,
                    config.stream.frame_rate);
            }
            if (recorder.recording()) {
                recorder.WriteFrame(grayscale.data, grayscale.total());
            }

            int key = -1;
            if (config.preview.enabled) {
                cv::imshow(kWindowName, grayscale);
                window_created = true;
                key = cv::waitKey(1) & 0xff;
            }
            if (key == 27 || key == 'q' || key == 'Q') {
                running.store(false);
            } else if (key == 'r' || key == 'R') {
                toggle_recording();
            }
        }

        recorder.Stop();
        wireless_controller.Stop();
        record_switch.Stop();
        viewer.Stop();
        if (window_created) {
            cv::destroyWindow(kWindowName);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Depth viewer failed: " << error.what() << std::endl;
        return 1;
    }
}
