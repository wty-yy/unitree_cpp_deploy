// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <deque>

namespace isaaclab
{

class ManagerBasedRLEnv;

using ObsFunc = std::function<std::vector<float>(ManagerBasedRLEnv*)>;

struct ObservationTermCfg
{
    ObsFunc func;
    std::vector<float> clip;
    std::vector<float> scale;
    int history_length = 1;

    void reset(std::vector<float> obs)
    {
        for(int i(0); i < history_length; ++i)
        {
            add(obs);
        }
    }

    void add(std::vector<float> obs)
    {
        buff_.push_back(obs);
        if (buff_.size() > history_length)
        {
            buff_.pop_front();
        }
    }

    // Complete circular buffer with most recent entry at the end and oldest entry at the beginning.
    std::vector<float> get()
    {
        // get observation
        std::vector<float> obs;
        for (int i = 0; i < buff_.size(); ++i)
        {
            auto obs_i = buff_[i];
            for(int j = 0; j < obs_i.size(); ++j)
            {
                if (!clip.empty()) {
                    obs_i[j] = std::clamp(obs_i[j], clip[0], clip[1]);
                }
                if(!scale.empty()) obs_i[j] *= scale[j];
            }
            obs.insert(obs.end(), obs_i.begin(), obs_i.end());
        }

        return obs;
    }

private:
    std::deque<std::vector<float>> buff_;
};

};