// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual std::map<std::string, std::vector<float>> forward(std::unordered_map<std::string, std::vector<float>> obs) { return {}; }
    virtual void reset() {}
    
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
    struct Options
    {
        bool enable_tensorrt = false;
        bool enable_cuda = false;
        int device_id = 0;
        std::string tensorrt_engine_cache_path;
    };

    explicit OrtRunner(std::string model_path)
        : OrtRunner(std::move(model_path), Options{})
    {
    }

    OrtRunner(std::string model_path, const Options& options)
        : env(ORT_LOGGING_LEVEL_WARNING, "onnx_model")
    {
        initialize_session(model_path, options);

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
            const auto size = tensor_size(shape, "input");
            input_sizes.push_back(size);
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

        recurrent_input_states_.resize(num_inputs);
        recurrent_output_to_input_.assign(num_outputs, no_index);
        for (size_t input_index = 0; input_index < num_inputs; ++input_index) {
            const auto& input_name = input_names_strings[input_index];
            if (!ends_with(input_name, "_in")) {
                continue;
            }

            const auto output_name = input_name.substr(0, input_name.size() - 3) + "_out";
            const auto output = std::find(
                output_names_strings.begin(), output_names_strings.end(), output_name);
            if (output == output_names_strings.end()) {
                continue;
            }

            const size_t output_index = std::distance(output_names_strings.begin(), output);
            const auto output_shape = session->GetOutputTypeInfo(output_index)
                .GetTensorTypeAndShapeInfo().GetShape();
            if (tensor_size(output_shape, "recurrent output") != input_sizes[input_index]) {
                throw std::runtime_error(
                    "Recurrent ONNX state size mismatch between '" + input_name +
                    "' and '" + output_name + "'.");
            }
            recurrent_input_states_[input_index].assign(input_sizes[input_index], 0.0f);
            recurrent_output_to_input_[output_index] = input_index;
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
    }

    const std::string& execution_provider() const
    {
        return execution_provider_;
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

    void reset() override
    {
        for (auto& state : recurrent_input_states_) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
    }

    std::map<std::string, std::vector<float>> forward(std::unordered_map<std::string, std::vector<float>> obs) override
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Make sure all model input names exist in obs
        for (size_t i = 0; i < input_names_strings.size(); ++i) {
            const auto& name = input_names_strings[i];
            if (obs.find(name) == obs.end() && recurrent_input_states_[i].empty()) {
                throw std::runtime_error("Input name '" + name + "' not found in observations or recurrent state.");
            }
        }

        // Create input tensors in model input order
        std::vector<Ort::Value> input_tensors;
        for(size_t i = 0; i < input_names.size(); ++i)
        {
            auto observation = obs.find(input_names_strings[i]);
            auto& input_data = observation != obs.end()
                ? observation->second
                : recurrent_input_states_[i];
            if (input_data.size() != input_sizes[i]) {
                throw std::runtime_error(
                    "Input '" + input_names_strings[i] + "' has " +
                    std::to_string(input_data.size()) + " values; expected " +
                    std::to_string(input_sizes[i]) + ".");
            }
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

            const auto recurrent_input_index = recurrent_output_to_input_[i];
            if (recurrent_input_index != no_index) {
                auto& state = recurrent_input_states_[recurrent_input_index];
                if (count != state.size()) {
                    throw std::runtime_error(
                        "Recurrent output '" + output_names_strings[i] +
                        "' changed size during inference.");
                }
                std::copy(floatarr, floatarr + count, state.begin());
            }
            
            std::vector<float> val(floatarr, floatarr + count);
            results[output_names_strings[i]] = val;
        }
        
        if(results.count("actions")) {
             std::lock_guard<std::mutex> lock(act_mtx_);
             action = results["actions"];
        }
        
        return results;
    }

private:
    struct ProviderCandidate
    {
        bool tensorrt = false;
        bool cuda = false;
    };

    static constexpr size_t no_index = std::numeric_limits<size_t>::max();

    static bool contains_provider(
        const std::vector<std::string>& providers, const std::string& name)
    {
        return std::find(providers.begin(), providers.end(), name) != providers.end();
    }

    static std::string provider_list(const std::vector<std::string>& providers)
    {
        if (providers.empty()) {
            return "none";
        }

        std::string result;
        for (const auto& provider : providers) {
            if (!result.empty()) {
                result += ", ";
            }
            result += provider;
        }
        return result;
    }

    static std::string candidate_name(const ProviderCandidate& candidate)
    {
        if (candidate.tensorrt && candidate.cuda) {
            return "TensorRT -> CUDA -> CPU";
        }
        if (candidate.tensorrt) {
            return "TensorRT -> CPU";
        }
        if (candidate.cuda) {
            return "CUDA -> CPU";
        }
        return "CPU";
    }

    void initialize_session(const std::string& model_path, const Options& options)
    {
        if (options.device_id < 0) {
            throw std::invalid_argument("ONNX Runtime device_id must be non-negative");
        }

        std::vector<std::string> available_providers;
        if (options.enable_tensorrt || options.enable_cuda) {
            try {
                available_providers = Ort::GetAvailableProviders();
                spdlog::info(
                    "ONNX Runtime available providers: {}",
                    provider_list(available_providers));
            } catch (const Ort::Exception& error) {
                spdlog::warn(
                    "Could not query ONNX Runtime providers: {}; falling back to CPU",
                    error.what());
            }
        }

        const bool has_tensorrt = contains_provider(
            available_providers, "TensorrtExecutionProvider");
        const bool has_cuda = contains_provider(
            available_providers, "CUDAExecutionProvider");
        if (options.enable_tensorrt && !has_tensorrt) {
            spdlog::warn("TensorRT execution provider is not available");
        }
        if (options.enable_cuda && !has_cuda) {
            spdlog::warn("CUDA execution provider is not available");
        }

        std::vector<ProviderCandidate> candidates;
        if (options.enable_tensorrt && has_tensorrt) {
            if (options.enable_cuda && has_cuda) {
                candidates.push_back({true, true});
            }
            candidates.push_back({true, false});
        }
        if (options.enable_cuda && has_cuda) {
            candidates.push_back({false, true});
        }
        candidates.push_back({false, false});

        std::string last_error;
        bool tensorrt_registration_failed = false;
        bool cuda_registration_failed = false;
        for (const auto& candidate : candidates) {
            if ((candidate.tensorrt && tensorrt_registration_failed) ||
                (candidate.cuda && cuda_registration_failed))
            {
                continue;
            }

            const auto name = candidate_name(candidate);
            Ort::SessionOptions candidate_options;
            candidate_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

            if (candidate.tensorrt) {
                try {
                    OrtTensorRTProviderOptions tensorrt_options{};
                    tensorrt_options.device_id = options.device_id;
                    tensorrt_options.trt_max_partition_iterations = 1000;
                    tensorrt_options.trt_min_subgraph_size = 1;
                    tensorrt_options.trt_max_workspace_size = 1ULL << 30;
                    tensorrt_options.trt_engine_cache_enable =
                        options.tensorrt_engine_cache_path.empty() ? 0 : 1;
                    tensorrt_options.trt_engine_cache_path =
                        options.tensorrt_engine_cache_path.empty()
                            ? nullptr
                            : options.tensorrt_engine_cache_path.c_str();
                    candidate_options.AppendExecutionProvider_TensorRT(tensorrt_options);
                } catch (const Ort::Exception& error) {
                    last_error = error.what();
                    tensorrt_registration_failed = true;
                    spdlog::warn(
                        "Could not register TensorRT execution provider: {}; "
                        "trying the next provider",
                        last_error);
                    continue;
                }
            }
            if (candidate.cuda) {
                try {
                    OrtCUDAProviderOptions cuda_options{};
                    cuda_options.device_id = options.device_id;
                    candidate_options.AppendExecutionProvider_CUDA(cuda_options);
                } catch (const Ort::Exception& error) {
                    last_error = error.what();
                    cuda_registration_failed = true;
                    spdlog::warn(
                        "Could not register CUDA execution provider: {}; "
                        "trying the next provider",
                        last_error);
                    continue;
                }
            }

            try {
                session = std::make_unique<Ort::Session>(
                    env, model_path.c_str(), candidate_options);
                execution_provider_ = name;
                spdlog::info(
                    "Loaded ONNX model {} with execution providers: {}",
                    model_path, execution_provider_);
                return;
            } catch (const Ort::Exception& error) {
                last_error = error.what();
                spdlog::warn(
                    "Could not create ONNX session with {}: {}",
                    name, last_error);
            }
        }

        throw std::runtime_error(
            "Could not create ONNX Runtime session for '" + model_path +
            "': " + last_error);
    }

    static bool ends_with(const std::string& value, const std::string& suffix)
    {
        return value.size() >= suffix.size()
            && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    static size_t tensor_size(const std::vector<int64_t>& shape, const char* description)
    {
        size_t size = 1;
        for (const auto dimension : shape) {
            if (dimension <= 0) {
                throw std::runtime_error(
                    std::string("Dynamic or invalid ") + description +
                    " dimension is not supported.");
            }
            size *= static_cast<size_t>(dimension);
        }
        return size;
    }

    Ort::Env env;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;
    std::string execution_provider_ = "CPU";

    std::vector<const char*> input_names;
    std::vector<std::string> input_names_strings;
    std::vector<const char*> output_names;
    std::vector<std::string> output_names_strings;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<size_t> input_sizes;
    std::vector<int64_t> output_shape;
    std::vector<std::vector<float>> recurrent_input_states_;
    std::vector<size_t> recurrent_output_to_input_;
};

};
