#include "FSM/CtrlFSM.h"
#include <atomic>
#include <iostream>
#include <thread>

std::unique_ptr<LowCmd_t> FSMState::lowcmd;
std::shared_ptr<LowState_t> FSMState::lowstate;
std::shared_ptr<Keyboard> FSMState::keyboard;

class EntryState : public BaseState {
public:
    EntryState(int id, const char* name, bool delay = false)
        : BaseState(id, name), value_(float(id)), delay_(delay) {}
    void enter() override {
        for (auto& motor : FSMState::lowcmd->msg_.motor_cmd()) motor.kp() = value_;
        // Make a mixed old-target/new-gain frame deterministic if enter() runs
        // outside the publication lock.
        if (delay_) std::this_thread::sleep_for(std::chrono::milliseconds(30));
        run();
        entered = true;
    }
    void run() override {
        for (auto& motor : FSMState::lowcmd->msg_.motor_cmd()) motor.q() = value_;
    }
    void post_run() override { FSMState::lowcmd->unlockAndPublish(); }
    std::atomic<bool> entered{false};
private:
    float value_;
    bool delay_;
};

int main() {
    try {
        // The controller is process-lived, as in production. All test commands
        // stay on loopback in a separate domain, with no robot connection.
        unitree::robot::ChannelFactory::Instance()->Init(158, "lo");
        FSMState::lowcmd = std::make_unique<LowCmd_t>();
        FSMState::lowstate = std::make_shared<LowState_t>();
        std::atomic<unsigned> received{0}, mixed{0};
        unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowCmd_> observer("rt/lowcmd");
        observer.InitChannel([&](const void* message) {
            const auto& msg = *static_cast<const unitree_hg::msg::dds_::LowCmd_*>(message);
            bool coherent = true;
            for (const auto& motor : msg.motor_cmd()) {
                coherent = coherent && motor.q() == motor.kp();
            }
            if (!coherent) ++mixed;
            ++received;
        }, 10);
        auto first = std::make_shared<EntryState>(1, "Passive");
        auto second = std::make_shared<EntryState>(2, "EntryTest", true);
        std::atomic<bool> transition{false};
        first->registered_checks.push_back({[&] { return transition.load(); }, 2});
        auto* fsm = new CtrlFSM(first);
        fsm->add(second);
        fsm->start();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (received < 10 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (received < 10) throw std::runtime_error("no loopback command frames received");
        transition = true;
        while (!second->entered && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (!second->entered) throw std::runtime_error("test state was not entered");
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        if (mixed != 0) throw std::runtime_error("published a mixed old-target/new-gain entry frame");
        std::cout << "PASS atomic entry publication: " << received.load() << " complete frames\n" << std::flush;
        std::_Exit(0);
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n' << std::flush;
        std::_Exit(1);
    }
}
