#pragma once

#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <unitree/idl/ros2/String_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

namespace utils
{
// Read-only telemetry: never publishes motor commands.
class PolicySelectionPublisher
{
public:
    explicit PolicySelectionPublisher(std::vector<std::string> expert_names)
        : expert_names_(std::move(expert_names)), publisher_("rt/go2/policy_selection")
    {
        publisher_.InitChannel();
    }

    void publish(const std::map<std::string, std::vector<float>>& results)
    {
        const auto selected = results.find("selected_id");
        if (selected == results.end() || selected->second.size() != 1) {
            clear();
            return;
        }
        const float id = selected->second[0];
        if (!std::isfinite(id) || id < 0 || id >= expert_names_.size() || std::floor(id) != id) {
            clear();
            return;
        }
        std_msgs::msg::dds_::String_ message;
        message.data(nlohmann::json{{"active", true}, {"selected_id", static_cast<int>(id)},
                                    {"selected_name", expert_names_[static_cast<std::size_t>(id)]}}.dump());
        publisher_.Write(message);
    }

    void clear()
    {
        std_msgs::msg::dds_::String_ message;
        message.data("{\"active\":false}");
        publisher_.Write(message);
    }

private:
    std::vector<std::string> expert_names_;
    unitree::robot::ChannelPublisher<std_msgs::msg::dds_::String_> publisher_;
};
}
