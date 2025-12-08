// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include "isaaclab/manager/manager_term_cfg.h"
#include <iostream>

namespace isaaclab
{

using ObsMap = std::map<std::string, ObsFunc>;

inline ObsMap& observations_map() {
    static ObsMap instance;
    return instance;
}

#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env)


class ObservationManager
{
public:
    ObservationManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    :cfg(cfg), env(env)
    {
        _prapare_terms();
    }

    void reset()
    {
        for(auto & term : obs_term_cfgs)
        {
            term.reset(term.func(this->env));
        }
    }

    std::vector<float> compute()
    {
        std::vector<float> obs;
        for(auto & term : obs_term_cfgs)
        {
            term.add(term.func(this->env));
            auto term_obs_scaled = term.get();
            obs.insert(obs.end(), term_obs_scaled.begin(), term_obs_scaled.end());
        }
        return obs;
    }

protected:
    void _prapare_terms()
    {
        for(auto it = this->cfg.begin(); it != this->cfg.end(); ++it)
        {
            auto term_yaml_cfg = it->second;
            ObservationTermCfg term_cfg;
            term_cfg.history_length = term_yaml_cfg["history_length"].as<int>(1);

            auto term_name = it->first.as<std::string>();
            if(observations_map()[term_name] == nullptr)
            {
                throw std::runtime_error("Observation term '" + term_name + "' is not registered.");
            }
            term_cfg.func = observations_map()[term_name];   

            auto obs = term_cfg.func(this->env);
            term_cfg.reset(obs);
            term_cfg.scale = term_yaml_cfg["scale"].as<std::vector<float>>();
            if(!term_yaml_cfg["clip"].IsNull()) {
                term_cfg.clip = term_yaml_cfg["clip"].as<std::vector<float>>();
            }

            this->obs_term_cfgs.push_back(term_cfg);
        }
    }

    YAML::Node cfg;
    ManagerBasedRLEnv* env;
private:
    std::vector<ObservationTermCfg> obs_term_cfgs;
};

};