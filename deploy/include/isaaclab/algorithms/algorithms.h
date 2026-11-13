// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <map>
#include <unordered_map>
#include <string>

namespace isaaclab
{

class Algorithms
{
public:
    virtual ~Algorithms() = default;
    virtual void reset() {}
    // Pay the one-off execution-provider cost (TensorRT compiles dynamic-shape
    // engines on the first Run) up front instead of on the first policy switch.
    virtual void warmup(int /*iterations*/ = 2) {}
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;
    virtual std::map<std::string, std::vector<float>> forward(std::unordered_map<std::string, std::vector<float>> obs) { return {}; }
    
    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }

    // Read by the policy thread; the logging snapshot carries it to other threads.
    double last_onnx_inference_ms() const { return last_onnx_inference_ms_; }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
    double last_onnx_inference_ms_ = std::numeric_limits<double>::quiet_NaN();
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path, const YAML::Node& cfg = YAML::Node())
        : model_path_(std::move(model_path))
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        configure_providers(model_path_, cfg);

        session = std::make_unique<Ort::Session>(env, model_path_.c_str(), session_options);

        // Dynamic input detection
        size_t num_inputs = session->GetInputCount();
        for (size_t i = 0; i < num_inputs; ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            // Exported actors can declare a symbolic batch; deployment runs one robot.
            auto& shape = input_shapes.back();
            if (!shape.empty() && shape[0] == -1) shape[0] = 1;
            for (const auto dim : shape) {
                if (dim <= 0) throw std::runtime_error("Unsupported dynamic ONNX input dimension");
            }
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names_strings.push_back(input_name.get());
        }
        for (const auto& name : input_names_strings) {
            input_names.push_back(name.c_str());
        }
        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) size *= dim;
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
            if (input_data.size() != static_cast<std::size_t>(input_sizes[i])) {
                throw std::runtime_error("Input size mismatch for " + input_names_strings[i]);
            }
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        const auto inference_start = std::chrono::steady_clock::now();
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
        last_onnx_inference_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inference_start).count();
        
        std::map<std::string, std::vector<float>> results;
        for(size_t i=0; i<output_tensors.size(); i++) {
            auto floatarr = output_tensors[i].GetTensorMutableData<float>();
            auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            auto shape = type_info.GetShape();
            size_t count = 1;
            for(auto s : shape) count *= s;
            
            std::vector<float> val(floatarr, floatarr + count);
            results[output_names_strings[i]] = val;
        }
        
        if(results.count("actions")) {
             std::lock_guard<std::mutex> lock(act_mtx_);
             action = results["actions"];
        }
        
        return results;
    }

    // Run the model with zero inputs of the declared shapes. TensorRT only builds
    // the dynamic-shape engine on the first Run, so doing this at startup (or from
    // the precompile tool) removes the build latency from policy switching.
    // Calls the base forward explicitly: derived runners need live sensors.
    void warmup(int iterations = 2) override
    {
        std::unordered_map<std::string, std::vector<float>> obs;
        obs.reserve(input_names_strings.size());
        for (std::size_t i = 0; i < input_names_strings.size(); ++i)
        {
            obs.emplace(input_names_strings[i], std::vector<float>(input_sizes[i], 0.0f));
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) OrtRunner::forward(obs);
        warmup_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();

        std::cout << "OrtRunner: warmup " << model_path_ << " iterations=" << iterations
                  << " total_ms=" << warmup_ms_ << std::endl;
    }

private:
    void configure_providers(const std::string& model_path, const YAML::Node& cfg)
    {
        const bool enable_tensorrt = cfg["onnx_tensorrt"].as<bool>(false);
        const bool enable_cuda = cfg["onnx_cuda"].as<bool>(enable_tensorrt);
        const int device_id = cfg["onnx_cuda_device"].as<int>(0);
        const auto providers = Ort::GetAvailableProviders();
        auto available = [&](const char* name) {
            return std::find(providers.begin(), providers.end(), name) != providers.end();
        };
        auto append = [&](OrtStatus* status, const char* provider) {
            if (!status) return true;
            std::cerr << "OrtRunner: " << provider << " unavailable: "
                      << Ort::GetApi().GetErrorMessage(status) << std::endl;
            Ort::GetApi().ReleaseStatus(status);
            return false;
        };

        bool tensorrt = false;
        if (enable_tensorrt && available("TensorrtExecutionProvider"))
        {
            const auto cache = (std::filesystem::absolute(model_path).parent_path() / "trt_cache").string();
            std::filesystem::create_directories(cache);
            OrtTensorRTProviderOptions options{};
            options.device_id = device_id;
            options.trt_max_partition_iterations = 1000;
            options.trt_min_subgraph_size = 1;
            options.trt_max_workspace_size = 1ULL << 30;
            options.trt_engine_cache_enable = 1;
            options.trt_engine_cache_path = cache.c_str();
            // Keep FP32 precision; acceleration must not implicitly enable FP16.
            tensorrt = append(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT(
                session_options, &options), "TensorRT");
        }
        if (enable_tensorrt && !tensorrt)
            std::cerr << "OrtRunner: requested TensorRT could not be enabled for " << model_path << std::endl;

        // Register CUDA after TensorRT for operators that TensorRT cannot execute.
        bool cuda = false;
        if (enable_cuda && available("CUDAExecutionProvider"))
        {
            OrtCUDAProviderOptions options{};
            options.device_id = device_id;
            cuda = append(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA(
                session_options, &options), "CUDA");
        }
        if (enable_cuda && !cuda)
            std::cerr << "OrtRunner: requested CUDA could not be enabled for " << model_path << std::endl;
        std::cout << "OrtRunner: " << model_path << " execution providers: "
                  << (tensorrt ? "TensorRT > " : "") << (cuda ? "CUDA > " : "") << "CPU" << std::endl;
    }

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string model_path_;
    double warmup_ms_ = std::numeric_limits<double>::quiet_NaN();

    std::vector<const char*> input_names;
    std::vector<std::string> input_names_strings;
    std::vector<const char*> output_names;
    std::vector<std::string> output_names_strings;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};

};
