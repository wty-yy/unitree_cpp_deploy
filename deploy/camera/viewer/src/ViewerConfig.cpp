#include "ViewerConfig.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

namespace camera_viewer
{

ViewerConfig LoadViewerConfig(const std::filesystem::path& path)
{
    const YAML::Node root = YAML::LoadFile(path.string());
    const YAML::Node dds = root["dds"];
    const YAML::Node stream = root["stream"];
    const YAML::Node preview = root["preview"];
    const YAML::Node recording = root["recording"];
    if (!dds || !stream || !preview || !recording) {
        throw std::invalid_argument(
            "config.yaml requires dds, stream, preview and recording sections");
    }

    ViewerConfig config;
    config.dds.domain_id = dds["domain_id"].as<int>(0);
    config.dds.interface = dds["interface"].as<std::string>("lo");
    config.dds.topic = dds["topic"].as<std::string>("rt/front_depth_observation");
    config.dds.control_topic = dds["control_topic"].as<std::string>(
        "cpp_viewer/record_switch");
    config.dds.wireless_controller_topic =
        dds["wireless_controller_topic"].as<std::string>("rt/wirelesscontroller");
    config.dds.queue_length = dds["queue_length"].as<int>(0);

    config.stream.width = stream["width"].as<int>(640);
    config.stream.height = stream["height"].as<int>(360);
    config.stream.encoding = stream["encoding"].as<std::string>("32FC1");
    config.stream.frame_rate = stream["frame_rate"].as<int>(15);
    config.stream.stale_timeout = stream["stale_timeout"].as<double>(1.0);

    config.preview.enabled = preview["enabled"].as<bool>(true);
    config.preview.max_depth = preview["max_depth"].as<float>(2.0F);

    config.recording.output_dir = recording["output_dir"].as<std::string>("videos");
    config.recording.codec = recording["codec"].as<std::string>("libx265");
    config.recording.preset = recording["preset"].as<std::string>("fast");
    config.recording.crf = recording["crf"].as<int>(26);

    if (config.dds.interface.empty() || config.dds.topic.empty() ||
        config.dds.control_topic.empty() || config.dds.wireless_controller_topic.empty() ||
        config.dds.queue_length < 0)
    {
        throw std::invalid_argument("DDS interface/topic must be set and queue_length non-negative");
    }
    if (config.stream.width <= 0 || config.stream.height <= 0 ||
        config.stream.width % 2 != 0 || config.stream.height % 2 != 0 ||
        config.stream.encoding.empty() || config.stream.frame_rate <= 0 ||
        !std::isfinite(config.stream.stale_timeout) || config.stream.stale_timeout <= 0.0)
    {
        throw std::invalid_argument(
            "stream dimensions must be positive/even and stream settings valid");
    }
    if (!std::isfinite(config.preview.max_depth) || config.preview.max_depth <= 0.0F) {
        throw std::invalid_argument("preview.max_depth must be finite and positive");
    }
    if (config.recording.output_dir.empty() || config.recording.codec.empty() ||
        config.recording.preset.empty() || config.recording.crf < 0 ||
        config.recording.crf > 51)
    {
        throw std::invalid_argument("recording configuration is invalid");
    }
    return config;
}

}  // namespace camera_viewer
