#pragma once

#include "Config.h"

#include <librealsense2/rs.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace d435i
{

struct CameraCalibration
{
    float depth_scale = 0.0F;
    float fx = 0.0F;
    float fy = 0.0F;
    float ppx = 0.0F;
    float ppy = 0.0F;
    float horizontal_fov = 0.0F;
    float vertical_fov = 0.0F;
    rs2_distortion distortion = RS2_DISTORTION_NONE;
};

struct DepthFrameInfo
{
    std::uint64_t source_frame_number = 0;
    double timestamp_seconds = 0.0;
};

class D435iCamera
{
public:
    D435iCamera(CameraConfig config, ValidationConfig validation);
    ~D435iCamera();

    D435iCamera(const D435iCamera&) = delete;
    D435iCamera& operator=(const D435iCamera&) = delete;

    void Start();
    void Stop();
    DepthFrameInfo WaitForFrame(std::vector<float>& metric_depth);

    const CameraCalibration& calibration() const { return calibration_; }
    const std::string& device_name() const { return device_name_; }
    const std::string& serial() const { return serial_; }

private:
    void ValidateCalibration() const;

    CameraConfig config_;
    ValidationConfig validation_;
    rs2::pipeline pipeline_;
    CameraCalibration calibration_;
    std::string device_name_;
    std::string serial_;
    bool running_ = false;
};

}  // namespace d435i
