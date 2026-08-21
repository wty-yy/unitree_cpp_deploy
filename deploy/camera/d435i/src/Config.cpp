#include "Config.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

namespace d435i
{

AppConfig LoadConfig(const std::filesystem::path& path)
{
    const YAML::Node root = YAML::LoadFile(path.string());
    const YAML::Node dds = root["dds"];
    const YAML::Node camera = root["camera"];
    const YAML::Node validation = root["validation"];
    if (!dds || !camera || !validation) {
        throw std::invalid_argument(
            "config.yaml requires dds, camera and validation sections");
    }

    AppConfig config;
    config.dds.domain_id = dds["domain_id"].as<int>(0);
    config.dds.interface = dds["interface"].as<std::string>("lo");
    config.dds.topic = dds["topic"].as<std::string>("rt/front_depth_observation");

    config.camera.serial = camera["serial"].as<std::string>("");
    config.camera.frame_id = camera["frame_id"].as<std::string>(
        "d435i_depth_optical_frame");
    config.camera.width = camera["width"].as<int>(640);
    config.camera.height = camera["height"].as<int>(360);
    config.camera.capture_fps = camera["capture_fps"].as<int>(30);
    config.camera.publish_hz = camera["publish_hz"].as<int>(15);
    config.camera.warmup_frames = camera["warmup_frames"].as<int>(30);
    config.camera.wait_timeout_ms = camera["wait_timeout_ms"].as<int>(2000);
    config.camera.emitter_enabled = camera["emitter_enabled"].as<bool>(true);
    config.camera.near_clip = camera["near_clip"].as<float>(0.16F);
    config.camera.far_clip = camera["far_clip"].as<float>(2.0F);

    config.validation.expected_horizontal_fov =
        validation["expected_horizontal_fov"].as<float>(89.0F);
    config.validation.expected_vertical_fov =
        validation["expected_vertical_fov"].as<float>(58.0F);
    config.validation.fov_tolerance = validation["fov_tolerance"].as<float>(3.0F);
    config.validation.principal_point_tolerance_px =
        validation["principal_point_tolerance_px"].as<float>(8.0F);

    if (config.dds.interface.empty() || config.dds.topic.empty()) {
        throw std::invalid_argument("DDS interface and topic must not be empty");
    }
    if (config.camera.frame_id.empty()) {
        throw std::invalid_argument("camera.frame_id must not be empty");
    }
    if (config.camera.width <= 0 || config.camera.height <= 0 ||
        config.camera.capture_fps <= 0 || config.camera.publish_hz <= 0 ||
        config.camera.publish_hz > config.camera.capture_fps ||
        config.camera.capture_fps % config.camera.publish_hz != 0)
    {
        throw std::invalid_argument(
            "camera dimensions and rates must be positive, and capture_fps must be "
            "an integer multiple of publish_hz");
    }
    if (config.camera.warmup_frames < 0 || config.camera.wait_timeout_ms <= 0) {
        throw std::invalid_argument(
            "camera.warmup_frames must be non-negative and wait_timeout_ms positive");
    }
    if (!std::isfinite(config.camera.near_clip) ||
        !std::isfinite(config.camera.far_clip) ||
        config.camera.near_clip < 0.0F ||
        config.camera.far_clip <= config.camera.near_clip)
    {
        throw std::invalid_argument("camera depth range is invalid");
    }
    if (!std::isfinite(config.validation.fov_tolerance) ||
        config.validation.fov_tolerance < 0.0F ||
        !std::isfinite(config.validation.principal_point_tolerance_px) ||
        config.validation.principal_point_tolerance_px < 0.0F)
    {
        throw std::invalid_argument("validation tolerances must be finite and non-negative");
    }
    return config;
}

}  // namespace d435i
