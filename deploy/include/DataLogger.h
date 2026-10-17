#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>

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

        if (first_write_) {
            // Write header
            bool first = true;
            for (const auto& key : headers_) {
                if (!first) file_ << ",";
                file_ << key;
                first = false;
            }
            file_ << "\n";
            first_write_ = false;
        }

        // Write data
        bool first = true;
        for (const auto& key : headers_) {
            if (!first) file_ << ",";
            if (data_.count(key)) {
                file_ << data_[key];
            }
            first = false;
        }
        file_ << "\n";
        
        data_.clear();
    }

private:
    std::string filename_;
    std::ofstream file_;
    std::map<std::string, std::string> data_;
    std::vector<std::string> headers_;
    bool first_write_ = true;
};
