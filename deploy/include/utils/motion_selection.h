#pragma once

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

class MotionSelection
{
public:
    MotionSelection(std::vector<std::filesystem::path> motion_files, std::size_t initial_index)
        : files_(std::move(motion_files))
    {
        if (!files_.empty())
        {
            index_ = std::min(initial_index, files_.size() - 1);
        }
    }

    bool empty() const
    {
        return files_.empty();
    }

    std::size_t size() const
    {
        return files_.size();
    }

    std::size_t index() const
    {
        return index_;
    }

    const std::filesystem::path& current() const
    {
        return files_.at(index_);
    }

    bool move_next()
    {
        if (files_.empty())
        {
            return false;
        }
        index_ = (index_ + 1) % files_.size();
        return true;
    }

    bool move_prev()
    {
        if (files_.empty())
        {
            return false;
        }
        index_ = (index_ + files_.size() - 1) % files_.size();
        return true;
    }

    void set_index(std::size_t index)
    {
        if (files_.empty())
        {
            index_ = 0;
            return;
        }
        index_ = std::min(index, files_.size() - 1);
    }

    const std::vector<std::filesystem::path>& files() const
    {
        return files_;
    }

    std::string render_compact_ui() const
    {
        if (files_.empty())
        {
            return "State_Mimic selection: no motion files.";
        }

        std::string line = "State_Mimic selection [" + std::to_string(index_ + 1) + " / " + std::to_string(files_.size()) + "]: ";
        for (std::size_t i = 0; i < files_.size(); ++i)
        {
            if (i > 0)
            {
                line += " | ";
            }
            const std::string name = files_[i].filename().string();
            if (i == index_)
            {
                line += "-> ";
                line += name;
            }
            else
            {
                line += name;
            }
        }
        return line;
    }

    std::string render_live_line() const
    {
        if (files_.empty())
        {
            return "\r\033[2KState_Mimic selection: no motion files.";
        }

        std::ostringstream oss;
        oss << "\r\033[2KState_Mimic selection [" << (index_ + 1) << " / " << files_.size() << "]: -> "
            << files_[index_].filename().string()
            << "    (UP/DOWN select, A confirm, B cancel)";
        return oss.str();
    }

private:
    std::vector<std::filesystem::path> files_;
    std::size_t index_{0};
};
