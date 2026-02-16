#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize FSM
    auto & joy = FSMState::lowstate->joystick;
    auto fsm = std::make_unique<CtrlFSM>(new State_Passive(FSMMode::Passive));
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return joy.LT.pressed && joy.A.on_pressed; }, // L2 + A
            (int)FSMMode::FixStand
        )
    );
    fsm->add(new State_FixStand(FSMMode::FixStand));

    // Helper lambda to check if a policy is configured (not null)
    auto check_policy_valid = [](std::string policy_key) -> bool {
        auto cfg = param::config["FSM"]["Velocity"];
        if (!cfg[policy_key] || cfg[policy_key].IsNull()) {
            return false;
        }
        return true;
    };

// Helper lambda to register velocity transitions
    auto register_velocity_transitions = [&](BaseState* state) {
        // Start + Up -> Velocity_Up
        state->registered_checks.emplace_back(
            std::make_pair(
                [&, check_policy_valid]()->bool{ 
                    return joy.start.on_pressed && joy.up.pressed && check_policy_valid("policy_dir_up"); 
                }, 
                FSMMode::Velocity_Up
            )
        );
        // Start + Down -> Velocity_Down
        state->registered_checks.emplace_back(
            std::make_pair(
                [&, check_policy_valid]()->bool{ 
                    return joy.start.on_pressed && joy.down.pressed && check_policy_valid("policy_dir_down"); 
                }, 
                FSMMode::Velocity_Down
            )
        );
        // Start + Left -> Velocity_Left
        state->registered_checks.emplace_back(
            std::make_pair(
                [&, check_policy_valid]()->bool{ 
                    return joy.start.on_pressed && joy.left.pressed && check_policy_valid("policy_dir_left"); 
                }, 
                FSMMode::Velocity_Left
            )
        );
        // Start + Right -> Velocity_Right
        state->registered_checks.emplace_back(
            std::make_pair(
                [&, check_policy_valid]()->bool{ 
                    return joy.start.on_pressed && joy.right.pressed && check_policy_valid("policy_dir_right"); 
                }, 
                FSMMode::Velocity_Right
            )
        );
    };

    // Register transitions for FixStand
    register_velocity_transitions(fsm->states.back());

    // Add RL states and register transitions for them as well
    std::vector<State_RLBase*> rl_states;
    rl_states.push_back(new State_RLBase(FSMMode::Velocity_Up, "Velocity_Up", "policy_dir_up", "Velocity"));
    rl_states.push_back(new State_RLBase(FSMMode::Velocity_Down, "Velocity_Down", "policy_dir_down", "Velocity"));
    rl_states.push_back(new State_RLBase(FSMMode::Velocity_Left, "Velocity_Left", "policy_dir_left", "Velocity"));
    rl_states.push_back(new State_RLBase(FSMMode::Velocity_Right, "Velocity_Right", "policy_dir_right", "Velocity"));

    for(auto state : rl_states) {
        register_velocity_transitions(state);
        fsm->add(state);
    }

    std::cout << "Press [L2 + A] to enter FixStand mode.\n";
    std::cout << "Then press [Start + Up/Down/Left/Right] to select and start a policy.\n";
    std::cout << "Press [L2 + Y] to toggle fixed command execution (if enabled in config).\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

