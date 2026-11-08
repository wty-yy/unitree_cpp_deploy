#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <condition_variable>
#include <iostream>
#include <thread>

namespace isaaclab {
REGISTER_OBSERVATION(entry_state)
{
    auto action = env->action_manager->action();
    return {env->robot->data.joint_pos[0], env->robot->data.joint_pos[1], action[0], action[1]};
}
}

class MeasuredRobot : public isaaclab::Articulation {
public:
    float position = .2f;
    void update() override {
        data.joint_pos << position, -.3f;
        data.joint_vel.setZero();
    }
};

class DelayedPolicy : public isaaclab::Algorithms {
public:
    void reset() override {
        std::lock_guard<std::mutex> lock(mutex_);
        started_ = released_ = false;
    }
    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) override {
        return forward(std::move(obs)).at("actions");
    }
    std::map<std::string, std::vector<float>> forward(
        std::unordered_map<std::string, std::vector<float>> obs) override {
        std::unique_lock<std::mutex> lock(mutex_);
        observation = obs.at("obs");
        started_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this] { return released_; });
        return {{"actions", {.4f, -.6f}}};
    }
    bool wait_started() {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(2), [this] { return started_; });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        condition_.notify_all();
    }
    std::vector<float> observation;
private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool started_ = false, released_ = false;
};

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        const auto config = YAML::Load(R"(
joint_ids_map: [0, 1]
step_dt: 0.02
default_joint_pos: [0.2, -0.3]
actions:
  JointPositionAction:
    joint_ids: null
    clip: null
    scale: [0.5, 0.25]
    offset: [0.1, -0.1]
observations:
  entry_state:
    params: {}
    clip: null
    scale: null
    history_length: 10
)");
        auto robot = std::make_shared<MeasuredRobot>();
        isaaclab::ManagerBasedRLEnv env(config, robot);
        auto policy = std::make_unique<DelayedPolicy>();
        auto* delayed = policy.get();
        env.alg = std::move(policy);
        for (int activation = 0; activation < 2; ++activation) {
            robot->position = .2f + activation * .1f;
            const auto baseline = env.get_policy_logging_snapshot().inference_seq;
            env.reset();
            std::thread inference([&] { env.step(); });
            const bool started = delayed->wait_started();
            bool held = true;
            for (int tick = 0; tick < 20; ++tick) {
                held = held && env.get_policy_action_after(baseline).empty();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            delayed->release();
            inference.join();
            require(started, "delayed inference did not start");
            require(held, "uninitialized or previous activation action escaped while inference was pending");
            const auto target = env.get_policy_action_after(baseline);
            require(target.size() == 2, "complete action not committed after inference");
            require(std::abs(target[0]-.3f) < 1e-6f && std::abs(target[1]+.25f) < 1e-6f,
                    "action scaling or offset changed during handoff");
            require(env.get_policy_action_after(baseline+1).empty(), "same inference accepted as a fresh activation");
            require(delayed->observation.size() == 40, "history dimension changed");
            for (int frame = 0; frame < 10; ++frame) {
                const auto i = frame*4;
                require(std::abs(delayed->observation[i]-robot->position) < 1e-6f &&
                        std::abs(delayed->observation[i+1]+.3f) < 1e-6f,
                        "entry history did not repeat measured joints");
                require(delayed->observation[i+2] == 0 && delayed->observation[i+3] == 0,
                        "last_action reset no longer matches training");
            }
        }
        std::cout << "PASS delayed first inference, re-entry, coherent action handoff and current-state history\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
