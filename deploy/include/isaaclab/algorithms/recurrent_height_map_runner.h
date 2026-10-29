#pragma once

#include "isaaclab/algorithms/algorithms.h"
#include <functional>

namespace isaaclab
{
// mixed-inherit ONNX owns the term-major history update and the LSTM transition.
class RecurrentHeightMapRunner : public OrtRunner
{
public:
    RecurrentHeightMapRunner(const std::string& path,
                            std::function<std::vector<float>()> height_map)
        : OrtRunner(path), height_map_(std::move(height_map))
    {
        reset();
    }

    void reset() override
    {
        history_.assign(980, 0.0f);
        hidden_.assign(512, 0.0f);
        cell_.assign(512, 0.0f);
    }

    std::map<std::string, std::vector<float>> forward(
        std::unordered_map<std::string, std::vector<float>> obs) override
    {
        obs["depth"] = height_map_();
        obs["obs_history_in"] = history_;
        obs["h_in"] = hidden_;
        obs["c_in"] = cell_;
        auto result = OrtRunner::forward(std::move(obs));
        history_ = result.at("obs_history_out");
        hidden_ = result.at("h_out");
        cell_ = result.at("c_out");
        return result;
    }

private:
    std::function<std::vector<float>()> height_map_;
    std::vector<float> history_, hidden_, cell_;
};
}
