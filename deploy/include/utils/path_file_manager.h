#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

class PathFileManager
{
public:
    static std::filesystem::path resolve_path(
        const std::filesystem::path& input,
        const std::filesystem::path& base_dir
    )
    {
        if (input.is_absolute())
        {
            return input.lexically_normal();
        }
        return (base_dir / input).lexically_normal();
    }

    static std::vector<std::filesystem::path> collect_csv_files(
        const std::vector<std::filesystem::path>& inputs,
        const std::filesystem::path& base_dir
    )
    {
        std::vector<std::filesystem::path> files;
        for (const auto& input : inputs)
        {
            const auto resolved = resolve_path(input, base_dir);
            collect_csv_files_impl(resolved, files);
        }

        std::set<std::string> unique_keys;
        std::vector<std::filesystem::path> deduped;
        deduped.reserve(files.size());
        for (const auto& path : files)
        {
            const auto normalized = path.lexically_normal();
            const std::string key = normalized.string();
            if (unique_keys.insert(key).second)
            {
                deduped.push_back(normalized);
            }
        }

        std::sort(deduped.begin(), deduped.end(), [](const auto& a, const auto& b) {
            return a.string() < b.string();
        });

        return deduped;
    }

private:
    static void collect_csv_files_impl(
        const std::filesystem::path& path,
        std::vector<std::filesystem::path>& out_files
    )
    {
        if (!std::filesystem::exists(path))
        {
            return;
        }

        if (std::filesystem::is_regular_file(path))
        {
            if (is_csv(path))
            {
                out_files.push_back(path);
            }
            return;
        }

        if (!std::filesystem::is_directory(path))
        {
            return;
        }

        for (const auto& entry : std::filesystem::recursive_directory_iterator(path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (is_csv(entry.path()))
            {
                out_files.push_back(entry.path());
            }
        }
    }

    static bool is_csv(const std::filesystem::path& path)
    {
        std::string ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return ext == ".csv";
    }
};
