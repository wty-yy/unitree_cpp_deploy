// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <mutex>
#include <map>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual void reset() {}
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual std::map<std::string, std::vector<float>> forward(std::unordered_map<std::string, std::vector<float>> obs) { return {}; }
    
    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        // Dynamic input detection
        size_t num_inputs = session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names_strings.push_back(input_name.get());
        }
        for (const auto& name : input_names_strings) {
            input_names.push_back(name.c_str());
        }
        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                if (dim <= 0) throw std::runtime_error("Deployment requires fixed positive ONNX dimensions.");
                size *= dim;
            }
            input_sizes.push_back(size);
        }
        const auto depth = std::find(input_names_strings.begin(), input_names_strings.end(), "depth");
        if (depth != input_names_strings.end()) {
            const auto channels = session->GetModelMetadata().LookupCustomMetadataMapAllocated("image_channel_names", allocator);
            if (!channels || std::string(channels.get()) != "relative_elevation,missing_mask")
                throw std::runtime_error("Height-map policy requires relative_elevation,missing_mask metadata.");
            const auto& image_shape = input_shapes[std::distance(input_names_strings.begin(), depth)];
            if (image_shape != std::vector<int64_t>{1, 2, 17, 11} && image_shape != std::vector<int64_t>{1, 2, 9, 11})
                throw std::runtime_error("Height-map ONNX depth must have shape [1,2,17,11] or [1,2,9,11].");
            const std::map<std::string, std::vector<int64_t>> expected{
                {"single_obs", {1, 45}}, {"depth", image_shape},
                {"obs_history_in", {1, 450}}, {"h_in", {2, 1, 256}}, {"c_in", {2, 1, 256}}
            };
            if (input_names_strings.size() != expected.size())
                throw std::runtime_error("Unexpected height-map policy inputs.");
            for (size_t i = 0; i < input_names_strings.size(); ++i) {
                const auto shape = expected.find(input_names_strings[i]);
                if (shape == expected.end() || shape->second != input_shapes[i])
                    throw std::runtime_error("Unexpected height-map input shape: " + input_names_strings[i]);
            }
        }

        // Dynamic output detection
        size_t num_outputs = session->GetOutputCount();
        output_names_strings.reserve(num_outputs);
        output_names.reserve(num_outputs);

        for(size_t i = 0; i < num_outputs; i++) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            output_names_strings.push_back(name.get());
        }
        
        for(size_t i = 0; i < num_outputs; i++) {
            output_names.push_back(output_names_strings[i].c_str());
        }

        // Find "actions" output shape for compatibility
        bool action_found = false;
        for(size_t i=0; i<num_outputs; i++) {
             if(output_names_strings[i] == "actions") {
                 auto output_type = session->GetOutputTypeInfo(i);
                 output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
                 action.resize(output_shape[1]);
                 action_found = true;
                 break;
             }
        }
        if (!action_found && num_outputs > 0) {
             auto output_type = session->GetOutputTypeInfo(0);
             output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
             action.resize(output_shape[1]);
        }
        for (size_t i = 0; i < input_names_strings.size(); ++i) {
            const auto& name = input_names_strings[i];
            if (name == "obs_history_in" || name == "h_in" || name == "c_in") {
                auto output = name.substr(0, name.size() - 3) + "_out";
                if (std::find(output_names_strings.begin(), output_names_strings.end(), output) == output_names_strings.end())
                    throw std::runtime_error("Missing recurrent output: " + output);
                recurrent_state_[name].resize(input_sizes[i], 0.0f);
            }
        }
    }

    void reset() override
    {
        for (auto& state : recurrent_state_) std::fill(state.second.begin(), state.second.end(), 0.0f);
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::fill(action.begin(), action.end(), 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        auto results = forward(obs);
        if (results.count("actions")) {
            return results["actions"];
        } else if (!results.empty()) {
            return results.begin()->second;
        }
        return {};
    }

    std::map<std::string, std::vector<float>> forward(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        for (const auto& state : recurrent_state_) obs[state.first] = state.second;

        // Make sure all model input names exist in obs
        for (const auto& name : input_names_strings) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name '" + name + "' not found in observations.");
            }
        }

        // Create input tensors in model input order
        std::vector<Ort::Value> input_tensors;
        for(size_t i = 0; i < input_names.size(); ++i)
        {
            auto& input_data = obs.at(input_names_strings[i]);
            if (input_data.size() != static_cast<size_t>(input_sizes[i]) ||
                !std::all_of(input_data.begin(), input_data.end(), [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("Invalid ONNX input: " + input_names_strings[i]);
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
        
        std::map<std::string, std::vector<float>> results;
        for(size_t i=0; i<output_tensors.size(); i++) {
            auto floatarr = output_tensors[i].GetTensorMutableData<float>();
            auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();
            size_t count = 1;
            for(auto s : shape) count *= s;
            
            std::vector<float> val(floatarr, floatarr + count);
            if (!std::all_of(val.begin(), val.end(), [](float value) { return std::isfinite(value); }))
                throw std::runtime_error("Non-finite ONNX output: " + output_names_strings[i]);
            results[output_names_strings[i]] = val;
        }
        for (auto& state : recurrent_state_) {
            const auto& next = results.at(state.first.substr(0, state.first.size() - 3) + "_out");
            if (next.size() != state.second.size()) throw std::runtime_error("Recurrent state shape mismatch.");
            state.second = next;
        }
        
        if(results.count("actions")) {
             std::lock_guard<std::mutex> lock(act_mtx_);
             action = results["actions"];
        }
        
        return results;
    }

private:
    std::unordered_map<std::string, std::vector<float>> recurrent_state_;
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<std::string> input_names_strings;
    std::vector<const char*> output_names;
    std::vector<std::string> output_names_strings;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};

};
