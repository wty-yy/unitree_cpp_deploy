#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <nlohmann/json.hpp>

class DataLogger {
public:
    DataLogger(const std::string& filename) : filename_(filename) {
        // Create directory if it doesn't exist
        std::filesystem::path path(filename);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        file_.open(filename);
    }

    ~DataLogger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    void add(const std::string& key, float value) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(4) << value;
        if (first_write_) {
            if (std::find(headers_.begin(), headers_.end(), key) == headers_.end()) {
                headers_.push_back(key);
            }
        }
        data_[key] = ss.str();
    }

    void add(const std::string& key, const std::string& value) {
        if (first_write_) {
            if (std::find(headers_.begin(), headers_.end(), key) == headers_.end()) {
                headers_.push_back(key);
            }
        }
        data_[key] = value;
    }

    void add(const std::string& key, const std::vector<float>& values) {
        for (size_t i = 0; i < values.size(); ++i) {
            std::string full_key = key + "_" + std::to_string(i);
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4) << values[i];
            if (first_write_) {
                if (std::find(headers_.begin(), headers_.end(), full_key) == headers_.end()) {
                    headers_.push_back(full_key);
                }
            }
            data_[full_key] = ss.str();
        }
    }

    void write() {
        if (!file_.is_open()) return;

        std::stringstream row;
        if (first_write_) {
            // Write header
            bool first = true;
            for (const auto& key : headers_) {
                if (!first) row << ",";
                row << key;
                first = false;
            }
            row << "\n";
            first_write_ = false;
        }

        // Write data
        bool first = true;
        for (const auto& key : headers_) {
            if (!first) row << ",";
            if (data_.count(key)) {
                row << data_[key];
            }
            first = false;
        }
        row << "\n";
        file_ << row.str();
        file_.flush();
        
        data_.clear();
    }

    void write_metadata(nlohmann::json metadata) {
        auto path = std::filesystem::path(filename_);
        metadata["events_file"] = path.replace_extension(".events.jsonl").filename().string();
        path = filename_;
        std::ofstream metadata_file(path.replace_extension(".metadata.json"));
        metadata_file << metadata.dump(2) << "\n";
    }

    std::size_t record_configuration(const std::string& kind, const nlohmann::json& value,
                                     double time, double unix_time) {
        const auto previous = configurations_.find(kind);
        if (previous != configurations_.end() && previous->second.value == value) {
            return previous->second.id;
        }
        const std::size_t id = previous == configurations_.end() ? 0 : previous->second.id + 1;
        if (!events_file_.is_open()) {
            auto path = std::filesystem::path(filename_);
            events_file_.open(path.replace_extension(".events.jsonl"));
        }
        events_file_ << nlohmann::json{
            {"kind", kind}, {"id", id}, {"time", time}, {"unix_time", unix_time}, {"values", value}
        }.dump() << "\n";
        events_file_.flush();
        configurations_[kind] = {value, id};
        return id;
    }

private:
    std::string filename_;
    std::ofstream file_;
    std::map<std::string, std::string> data_;
    std::vector<std::string> headers_;
    bool first_write_ = true;
    struct Configuration {
        nlohmann::json value;
        std::size_t id;
    };
    std::ofstream events_file_;
    std::map<std::string, Configuration> configurations_;
};
