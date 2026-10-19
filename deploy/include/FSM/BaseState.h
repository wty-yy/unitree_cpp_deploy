// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <boost/bimap.hpp>
#include <string>
#include <any>
#include <utility>
#include <type_traits>
#include <unordered_map>
#include <functional>
#include <memory>
#include <yaml-cpp/yaml.h>

inline boost::bimap<int, std::string> FSMStringMap;

class BaseState
{
public:
    BaseState(int state, std::string state_string) : state_(state) 
    {
        FSMStringMap.insert({state, state_string});
    }

    virtual void enter() {}

    virtual void pre_run() {}
    virtual void run() {}
    virtual void post_run() {}

    virtual void exit() {}

    std::string getStateString() { return FSMStringMap.left.at(state_); }
    int getState() {return state_; }
    bool isState(int state) { return state_ == state; }
    std::vector<std::pair<std::function<bool()>, int>> registered_checks;
private:
    int state_;
};

struct FsmPreflightResult
{
    bool enabled{true};
    std::string reason{};
};

using FsmFactory = std::function<std::shared_ptr<BaseState>(int, std::string)>;
using FsmPreflight = std::function<FsmPreflightResult(const YAML::Node&, const std::string&)>;

struct FsmRegistration
{
    FsmFactory factory;
    FsmPreflight preflight;
};

using FsmMap = std::unordered_map<std::string, FsmRegistration>;

inline FsmMap& getFsmMap() {
    static FsmMap fsmMap;
    return fsmMap;
}

template <typename T, typename = void>
struct has_fsm_preflight : std::false_type {};

template <typename T>
struct has_fsm_preflight<T, std::void_t<decltype(T::preflight(std::declval<YAML::Node>(), std::declval<std::string>()))>>
    : std::true_type {};

template <typename T>
inline FsmPreflight make_fsm_preflight()
{
    if constexpr (has_fsm_preflight<T>::value)
    {
        return [](const YAML::Node& cfg, const std::string& state_name) {
            return T::preflight(cfg, state_name);
        };
    }
    else
    {
        return [](const YAML::Node&, const std::string&) {
            return FsmPreflightResult{};
        };
    }
}

#define REGISTER_FSM(Derived) \
    inline std::shared_ptr<BaseState> __factory_##Derived(int s, std::string ss) {      \
        return std::make_shared<Derived>(s, ss);                                        \
    }                                                                                   \
    inline struct __registrar_##Derived {                                               \
        __registrar_##Derived() {                                                       \
            getFsmMap()[#Derived] = FsmRegistration{__factory_##Derived, make_fsm_preflight<Derived>()}; \
        }                                                                               \
    } __registrar_instance_##Derived;
