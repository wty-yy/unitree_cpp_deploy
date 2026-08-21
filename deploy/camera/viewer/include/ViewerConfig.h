#pragma once

#include <filesystem>
#include <string>

namespace camera_viewer
{

struct DdsConfig
{
    int domain_id = 0;
    std::string interface = "lo";
    std::string topic = "rt/front_depth_observation";
    std::string control_topic = "cpp_viewer/record_switch";
    std::string wireless_controller_topic = "rt/wirelesscontroller";
    int queue_length = 0;
};

struct StreamConfig
{
    int width = 640;
    int height = 360;
    std::string encoding = "32FC1";
    int frame_rate = 15;
    double stale_timeout = 1.0;
};

struct PreviewConfig
{
    bool enabled = true;
    float max_depth = 2.0F;
};

struct RecordingConfig
{
    std::filesystem::path output_dir = "videos";
    std::string codec = "libx265";
    std::string preset = "fast";
    int crf = 26;
};

struct ViewerConfig
{
    DdsConfig dds;
    StreamConfig stream;
    PreviewConfig preview;
    RecordingConfig recording;
};

ViewerConfig LoadViewerConfig(const std::filesystem::path& path);

}  // namespace camera_viewer
