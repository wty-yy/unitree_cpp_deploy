#pragma once

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <map>
#include <string>

namespace robot_automation {

// Describe only registered states and the policy paths their constructors resolved.
inline nlohmann::json make_policy_catalog(
    const YAML::Node& cfg, const nlohmann::json& policies,
    const std::map<std::string, std::string>& disabled = {},
    const nlohmann::json& overlays = nlohmann::json::array()) {
    using Json = nlohmann::json;
    Json states = Json::array();
    const auto registered = cfg["_"];
    for (const auto& entry : registered) {
        const auto name = entry.first.as<std::string>();
        const auto state = cfg[name];
        const auto reason = disabled.find(name);
        const std::string path = policies.value(name, std::string{});
        Json transitions = Json::array();
        if (state && state["transitions"]) {
            for (const auto& transition : state["transitions"]) {
                const auto target = transition.first.as<std::string>();
                if (registered[target]) transitions.push_back({
                    {"target", target}, {"expression", transition.second.as<std::string>()}});
            }
        }
        states.push_back({{"name", name},
            {"type", entry.second["type"] ? entry.second["type"].as<std::string>() : name},
            {"enabled", reason == disabled.end()},
            {"reason", reason == disabled.end() ? "" : reason->second},
            {"configured_policy_dir", state && state["policy_dir"] && !state["policy_dir"].IsNull() ? state["policy_dir"].as<std::string>() : ""},
            {"policy_dir", path}, {"policy_name", std::filesystem::path(path).filename().string()},
            {"transitions", transitions}});
    }
    return {{"schema_version", 1}, {"states", states}, {"overlays", overlays}};
}

} // namespace robot_automation
