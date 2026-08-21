#include "FfmpegRecorder.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef FFMPEG_EXECUTABLE_PATH
#define FFMPEG_EXECUTABLE_PATH "ffmpeg"
#endif

namespace camera_viewer
{

FfmpegRecorder::FfmpegRecorder(RecordingConfig config)
    : config_(std::move(config))
{
}

FfmpegRecorder::~FfmpegRecorder()
{
    Stop();
}

void FfmpegRecorder::Start(
    const std::filesystem::path& output_path,
    int width,
    int height,
    int frame_rate)
{
    if (recording()) {
        throw std::runtime_error("FFmpeg recorder is already running");
    }

    int descriptors[2];
    if (pipe(descriptors) != 0) {
        throw std::runtime_error(
            std::string("Could not create FFmpeg pipe: ") + std::strerror(errno));
    }

    const std::string video_size = std::to_string(width) + "x" + std::to_string(height);
    const std::string frame_rate_string = std::to_string(frame_rate);
    const std::string crf_string = std::to_string(config_.crf);
    output_path_ = output_path;
    expected_frame_size_ = static_cast<std::size_t>(width) * height;

    process_id_ = fork();
    if (process_id_ < 0) {
        close(descriptors[0]);
        close(descriptors[1]);
        process_id_ = -1;
        throw std::runtime_error(
            std::string("Could not fork FFmpeg: ") + std::strerror(errno));
    }

    if (process_id_ == 0) {
        setpgid(0, 0);
        dup2(descriptors[0], STDIN_FILENO);
        close(descriptors[0]);
        close(descriptors[1]);

        std::vector<std::string> arguments = {
            FFMPEG_EXECUTABLE_PATH,
            "-hide_banner",
            "-loglevel", "warning",
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "gray",
            "-video_size", video_size,
            "-framerate", frame_rate_string,
            "-i", "pipe:0",
            "-an",
            "-c:v", config_.codec,
            "-preset", config_.preset,
            "-crf", crf_string,
            "-x265-params", "log-level=error",
            "-pix_fmt", "yuv420p",
            output_path_.string(),
        };
        std::vector<char*> argv;
        argv.reserve(arguments.size() + 1);
        for (auto& argument : arguments) {
            argv.push_back(argument.data());
        }
        argv.push_back(nullptr);
        execv(FFMPEG_EXECUTABLE_PATH, argv.data());
        _exit(127);
    }

    close(descriptors[0]);
    input_fd_ = descriptors[1];
    std::cout << "Recording depth video: " << output_path_ << std::endl;
}

void FfmpegRecorder::WriteFrame(const unsigned char* grayscale, std::size_t size)
{
    if (!recording()) {
        return;
    }
    if (grayscale == nullptr || size != expected_frame_size_) {
        throw std::invalid_argument("Unexpected grayscale frame size for FFmpeg");
    }

    std::size_t written = 0;
    while (written < size) {
        const ssize_t result = write(input_fd_, grayscale + written, size - written);
        if (result > 0) {
            written += static_cast<std::size_t>(result);
            continue;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        throw std::runtime_error(
            std::string("Could not write frame to FFmpeg: ") + std::strerror(errno));
    }
}

bool FfmpegRecorder::Stop()
{
    if (!recording()) {
        return true;
    }

    close(input_fd_);
    input_fd_ = -1;
    int status = 0;
    pid_t result = -1;
    do {
        result = waitpid(process_id_, &status, 0);
    } while (result < 0 && errno == EINTR);
    process_id_ = -1;
    expected_frame_size_ = 0;
    const bool success = result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    if (success) {
        std::cout << "Saved depth video: " << output_path_ << std::endl;
    } else {
        std::cerr << "[WARN] FFmpeg exited unsuccessfully for " << output_path_ << std::endl;
    }
    return success;
}

}  // namespace camera_viewer
