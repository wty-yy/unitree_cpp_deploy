// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <mutex>
#include <map>
#include <string>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::vector<float> obs) = 0;
    virtual std::map<std::string, std::vector<float>> forward(std::vector<float> obs) { return {}; }
    
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

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        Ort::TypeInfo input_type = session->GetInputTypeInfo(0);
        input_shape = input_type.GetTensorTypeAndShapeInfo().GetShape();
        
        // Dynamic output detection
        size_t num_outputs = session->GetOutputCount();
        output_names.clear();
        output_names_strings.clear();
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

    std::vector<float> act(std::vector<float> obs) override
    {
        auto results = forward(obs);
        if (results.count("actions")) {
            return results["actions"];
        } else if (!results.empty()) {
            return results.begin()->second;
        }
        return {};
    }

    std::map<std::string, std::vector<float>> forward(std::vector<float> obs) override
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());
        
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
        
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

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    const std::vector<const char*> input_names = {"obs"};
    std::vector<const char*> output_names;
    std::vector<std::string> output_names_strings;

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
};

};