#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace deploy
{

// Downstream-only processing for the policy input. The simulator publishes the
// original metric image; Process() starts with a clone and never mutates it.
class D435iDepthProcessor
{
public:
    struct Config
    {
        int raw_width = 640;
        int raw_height = 360;
        int output_width = 56;
        int output_height = 32;
        float min_depth = 0.16F;
        float max_depth = 2.0F;
        int gaussian_kernel_size = 5;
        double gaussian_sigma = 1.0;
    };

    D435iDepthProcessor() : D435iDepthProcessor(Config{}) {}

    explicit D435iDepthProcessor(Config config) : config_(config)
    {
        if (config_.raw_width <= 0 || config_.raw_height <= 0 ||
            config_.output_width <= 0 || config_.output_height <= 0)
        {
            throw std::invalid_argument("Depth image dimensions must be positive");
        }
        if (config_.min_depth < 0.0F || config_.max_depth <= config_.min_depth)
        {
            throw std::invalid_argument("Depth range must satisfy 0 <= min_depth < max_depth");
        }
        if (config_.gaussian_kernel_size <= 0 || config_.gaussian_kernel_size % 2 == 0)
        {
            throw std::invalid_argument("Gaussian kernel size must be a positive odd number");
        }
    }

    // Returns contiguous normalized float data in [1, output_height, output_width] order.
    std::vector<float> Process(const std::vector<float>& metric_depth,
                               std::uint32_t width, std::uint32_t height) const
    {
        if (width != static_cast<std::uint32_t>(config_.raw_width) ||
            height != static_cast<std::uint32_t>(config_.raw_height) ||
            metric_depth.size() != static_cast<std::size_t>(width) * height)
        {
            throw std::invalid_argument("Unexpected raw D435i depth dimensions");
        }

        const cv::Mat input(config_.raw_height, config_.raw_width, CV_32FC1,
                            const_cast<float*>(metric_depth.data()));
        cv::Mat depth = input.clone();
        for (int row = 0; row < depth.rows; ++row)
        {
            float* pixels = depth.ptr<float>(row);
            for (int column = 0; column < depth.cols; ++column)
            {
                float& value = pixels[column];
                if (!std::isfinite(value) || value < config_.min_depth ||
                    value > config_.max_depth)
                {
                    value = config_.max_depth;
                }
            }
        }

        cv::resize(depth, depth,
                   cv::Size(config_.output_width, config_.output_height),
                   0.0, 0.0, cv::INTER_AREA);

        cv::GaussianBlur(depth, depth,
                         cv::Size(config_.gaussian_kernel_size, config_.gaussian_kernel_size),
                         config_.gaussian_sigma, config_.gaussian_sigma, cv::BORDER_REPLICATE);
        cv::max(depth, 0.0F, depth);
        cv::min(depth, config_.max_depth, depth);
        depth /= config_.max_depth;

        const float* begin = depth.ptr<float>(0);
        return std::vector<float>(begin, begin + depth.total());
    }

    const Config& config() const { return config_; }

private:
    Config config_;
};

}  // namespace deploy
