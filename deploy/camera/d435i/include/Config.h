#pragma once

#include <filesystem>
#include <string>

namespace d435i
{

struct DdsConfig
{
    int domain_id = 0;
    std::string interface = "lo";
    std::string topic = "rt/front_depth_observation";
};

struct CameraConfig
{
    std::string serial;
    std::string frame_id = "d435i_depth_optical_frame";
    int width = 640;
    int height = 360;
    int capture_fps = 30;
    int publish_hz = 15;
    int warmup_frames = 30;
    int wait_timeout_ms = 2000;
    bool emitter_enabled = true;
    float near_clip = 0.16F;
    float far_clip = 2.0F;
};

struct ValidationConfig
{
    float expected_horizontal_fov = 89.0F;
    float expected_vertical_fov = 58.0F;
    float fov_tolerance = 3.0F;
    float principal_point_tolerance_px = 8.0F;
};

struct AppConfig
{
    DdsConfig dds;
    CameraConfig camera;
    ValidationConfig validation;
};

AppConfig LoadConfig(const std::filesystem::path& path);

}  // namespace d435i
