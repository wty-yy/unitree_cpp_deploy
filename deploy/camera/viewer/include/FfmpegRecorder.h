#pragma once

#include "ViewerConfig.h"

#include <filesystem>
#include <sys/types.h>

namespace camera_viewer
{

class FfmpegRecorder
{
public:
    explicit FfmpegRecorder(RecordingConfig config);
    ~FfmpegRecorder();

    FfmpegRecorder(const FfmpegRecorder&) = delete;
    FfmpegRecorder& operator=(const FfmpegRecorder&) = delete;

    void Start(const std::filesystem::path& output_path, int width, int height, int frame_rate);
    void WriteFrame(const unsigned char* grayscale, std::size_t size);
    bool Stop();

    bool recording() const { return process_id_ > 0; }
    const std::filesystem::path& output_path() const { return output_path_; }

private:
    RecordingConfig config_;
    std::filesystem::path output_path_;
    int input_fd_ = -1;
    pid_t process_id_ = -1;
    std::size_t expected_frame_size_ = 0;
};

}  // namespace camera_viewer
