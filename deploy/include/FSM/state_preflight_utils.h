#pragma once

#include "FSM/BaseState.h"
#include "param.h"
#include "utils/path_file_manager.h"
#include <filesystem>
#include <string>
#include <vector>

namespace fsm_preflight
{

inline FsmPreflightResult disabled(std::string reason)
{
    return FsmPreflightResult{false, std::move(reason)};
}

inline std::filesystem::path resolve_under_project(const std::filesystem::path& path)
{
    if (path.is_absolute())
    {
        return path.lexically_normal();
    }
    return (param::proj_dir / path).lexically_normal();
}

inline std::filesystem::path resolve_under_policy(
    const std::filesystem::path& policy_dir,
    const std::filesystem::path& path
)
{
    if (path.is_absolute())
    {
        return path.lexically_normal();
    }
    return (policy_dir / path).lexically_normal();
}

inline FsmPreflightResult require_node(const YAML::Node& cfg, const std::string& key, const std::string& state_name)
{
    if (!cfg[key] || cfg[key].IsNull())
    {
        return disabled("FSM." + state_name + "." + key + " is missing");
    }
    return FsmPreflightResult{};
}

inline FsmPreflightResult require_file(const std::filesystem::path& path, const std::string& label)
{
    if (!std::filesystem::exists(path))
    {
        return disabled(label + " not found: " + path.string());
    }
    if (!std::filesystem::is_regular_file(path))
    {
        return disabled(label + " is not a regular file: " + path.string());
    }
    return FsmPreflightResult{};
}

inline FsmPreflightResult require_dir(const std::filesystem::path& path, const std::string& label)
{
    if (!std::filesystem::exists(path))
    {
        return disabled(label + " not found: " + path.string());
    }
    if (!std::filesystem::is_directory(path))
    {
        return disabled(label + " is not a directory: " + path.string());
    }
    return FsmPreflightResult{};
}

inline FsmPreflightResult require_policy_dir(
    const YAML::Node& cfg,
    const std::string& state_name,
    std::filesystem::path& out_policy_dir
)
{
    auto result = require_node(cfg, "policy_dir", state_name);
    if (!result.enabled)
    {
        return result;
    }

    out_policy_dir = resolve_under_project(cfg["policy_dir"].as<std::string>());
    result = require_dir(out_policy_dir, "policy_dir for " + state_name);
    if (!result.enabled)
    {
        return result;
    }

    out_policy_dir = param::parser_policy_dir(out_policy_dir);
    return require_dir(out_policy_dir, "resolved policy_dir for " + state_name);
}

inline std::vector<std::filesystem::path> configured_paths(const YAML::Node& node)
{
    std::vector<std::filesystem::path> paths;
    if (!node)
    {
        return paths;
    }
    if (node.IsScalar())
    {
        paths.emplace_back(node.as<std::string>());
    }
    else if (node.IsSequence())
    {
        for (const auto& item : node)
        {
            paths.emplace_back(item.as<std::string>());
        }
    }
    return paths;
}

inline FsmPreflightResult require_csv_motions(const YAML::Node& cfg, const std::string& state_name)
{
    const auto paths = configured_paths(cfg["motion_files"]);
    if (paths.empty())
    {
        return disabled("FSM." + state_name + ".motion_files is missing or empty");
    }

    const auto files = PathFileManager::collect_csv_files(paths, param::proj_dir);
    if (files.empty())
    {
        return disabled("no csv motion files found from FSM." + state_name + ".motion_files");
    }
    return FsmPreflightResult{};
}

inline FsmPreflightResult require_npz_motions(
    const YAML::Node& cfg,
    const std::filesystem::path& policy_dir,
    const std::string& state_name
)
{
    if (cfg["motion_files"] && cfg["motion_files"].IsSequence())
    {
        const auto paths = cfg["motion_files"].as<std::vector<std::string>>();
        if (paths.empty())
        {
            return disabled("FSM." + state_name + ".motion_files is empty");
        }
        for (const auto& path : paths)
        {
            const auto resolved = resolve_under_policy(policy_dir, path);
            auto result = require_file(resolved, "motion file for " + state_name);
            if (!result.enabled)
            {
                return result;
            }
        }
        return FsmPreflightResult{};
    }

    const auto motions_rel = cfg["motions_dir"] ? cfg["motions_dir"].as<std::string>() : "exported/motions";
    const auto motions_dir = resolve_under_policy(policy_dir, motions_rel);
    auto result = require_dir(motions_dir, "motions_dir for " + state_name);
    if (!result.enabled)
    {
        return result;
    }

    for (const auto& entry : std::filesystem::directory_iterator(motions_dir))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".npz")
        {
            return FsmPreflightResult{};
        }
    }
    return disabled("no npz motion files found in " + motions_dir.string());
}

} // namespace fsm_preflight
