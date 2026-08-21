#include "D435iCamera.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace d435i
{

D435iCamera::D435iCamera(CameraConfig config, ValidationConfig validation)
    : config_(std::move(config)), validation_(std::move(validation))
{
}

D435iCamera::~D435iCamera()
{
    Stop();
}

void D435iCamera::Start()
{
    if (running_) {
        return;
    }

    rs2::context context;
    const rs2::device_list devices = context.query_devices();
    if (devices.size() == 0) {
        throw std::runtime_error("No RealSense device detected");
    }

    std::string selected_serial = config_.serial;
    if (selected_serial.empty()) {
        selected_serial = devices.front().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    }

    rs2::config pipeline_config;
    pipeline_config.enable_device(selected_serial);
    pipeline_config.enable_stream(
        RS2_STREAM_DEPTH,
        config_.width,
        config_.height,
        RS2_FORMAT_Z16,
        config_.capture_fps);

    const rs2::pipeline_profile profile = pipeline_.start(pipeline_config);
    running_ = true;

    const rs2::device device = profile.get_device();
    device_name_ = device.supports(RS2_CAMERA_INFO_NAME)
        ? device.get_info(RS2_CAMERA_INFO_NAME)
        : "RealSense";
    serial_ = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);

    rs2::depth_sensor depth_sensor = device.first<rs2::depth_sensor>();
    calibration_.depth_scale = depth_sensor.get_depth_scale();
    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
        depth_sensor.set_option(
            RS2_OPTION_EMITTER_ENABLED,
            config_.emitter_enabled ? 1.0F : 0.0F);
    } else if (config_.emitter_enabled) {
        std::cerr << "[WARN] D435i depth sensor does not expose emitter control" << std::endl;
    }

    const rs2::video_stream_profile depth_profile =
        profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    if (depth_profile.width() != config_.width ||
        depth_profile.height() != config_.height ||
        depth_profile.fps() != config_.capture_fps ||
        depth_profile.format() != RS2_FORMAT_Z16)
    {
        throw std::runtime_error("RealSense selected an unexpected depth stream profile");
    }

    const rs2_intrinsics intrinsics = depth_profile.get_intrinsics();
    float fov[2] = {0.0F, 0.0F};
    rs2_fov(&intrinsics, fov);
    calibration_.fx = intrinsics.fx;
    calibration_.fy = intrinsics.fy;
    calibration_.ppx = intrinsics.ppx;
    calibration_.ppy = intrinsics.ppy;
    calibration_.horizontal_fov = fov[0];
    calibration_.vertical_fov = fov[1];
    calibration_.distortion = intrinsics.model;
    ValidateCalibration();

    std::cout << "RealSense device: " << device_name_ << " (serial " << serial_ << ")\n"
              << "Depth profile: " << config_.width << "x" << config_.height << "@"
              << config_.capture_fps << " Z16\n"
              << "Depth scale: " << calibration_.depth_scale << " m/unit\n"
              << "Intrinsics: fx=" << calibration_.fx << ", fy=" << calibration_.fy
              << ", ppx=" << calibration_.ppx << ", ppy=" << calibration_.ppy << "\n"
              << "Measured FOV: " << calibration_.horizontal_fov << "x"
              << calibration_.vertical_fov << " deg, distortion="
              << rs2_distortion_to_string(calibration_.distortion) << std::endl;
}

void D435iCamera::Stop()
{
    if (!running_) {
        return;
    }
    try {
        pipeline_.stop();
    } catch (const rs2::error& error) {
        std::cerr << "[WARN] Could not stop RealSense pipeline cleanly: "
                  << error.what() << std::endl;
    }
    running_ = false;
}

DepthFrameInfo D435iCamera::WaitForFrame(std::vector<float>& metric_depth)
{
    if (!running_) {
        throw std::runtime_error("RealSense pipeline is not running");
    }

    const rs2::frameset frames = pipeline_.wait_for_frames(config_.wait_timeout_ms);
    const rs2::depth_frame depth = frames.get_depth_frame();
    if (!depth) {
        throw std::runtime_error("RealSense frameset did not contain a depth frame");
    }
    if (depth.get_width() != config_.width || depth.get_height() != config_.height) {
        throw std::runtime_error("RealSense depth frame dimensions changed at runtime");
    }

    const auto* source = static_cast<const std::uint16_t*>(depth.get_data());
    const std::size_t pixel_count =
        static_cast<std::size_t>(config_.width) * config_.height;
    metric_depth.resize(pixel_count);
    for (std::size_t index = 0; index < pixel_count; ++index) {
        metric_depth[index] = static_cast<float>(source[index]) * calibration_.depth_scale;
    }

    DepthFrameInfo info;
    info.source_frame_number = depth.get_frame_number();
    info.timestamp_seconds = depth.get_timestamp() * 0.001;
    return info;
}

void D435iCamera::ValidateCalibration() const
{
    const float horizontal_error = std::abs(
        calibration_.horizontal_fov - validation_.expected_horizontal_fov);
    const float vertical_error = std::abs(
        calibration_.vertical_fov - validation_.expected_vertical_fov);
    if (horizontal_error > validation_.fov_tolerance ||
        vertical_error > validation_.fov_tolerance)
    {
        std::cerr << "[WARN] RealSense FOV differs from the policy calibration: measured "
                  << calibration_.horizontal_fov << "x" << calibration_.vertical_fov
                  << " deg, expected " << validation_.expected_horizontal_fov << "x"
                  << validation_.expected_vertical_fov << " deg" << std::endl;
    }

    const float center_x = static_cast<float>(config_.width) * 0.5F;
    const float center_y = static_cast<float>(config_.height) * 0.5F;
    if (std::abs(calibration_.ppx - center_x) >
            validation_.principal_point_tolerance_px ||
        std::abs(calibration_.ppy - center_y) >
            validation_.principal_point_tolerance_px)
    {
        std::cerr << "[WARN] RealSense principal point is far from the image center: ("
                  << calibration_.ppx << ", " << calibration_.ppy << ") vs ("
                  << center_x << ", " << center_y << ")" << std::endl;
    }
}

}  // namespace d435i
