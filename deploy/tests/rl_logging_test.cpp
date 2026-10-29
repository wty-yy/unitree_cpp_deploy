#include "FSM/State_RLBase.h"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

std::unique_ptr<LowCmd_t> FSMState::lowcmd;
std::shared_ptr<LowState_t> FSMState::lowstate;
std::shared_ptr<Keyboard> FSMState::keyboard;

namespace {
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::vector<std::vector<std::string>> rows;
    for (std::string line; std::getline(input, line);) {
        std::stringstream stream(line);
        std::vector<std::string> row;
        for (std::string field; std::getline(stream, field, ',');) row.push_back(field);
        rows.push_back(std::move(row));
    }
    return rows;
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "Usage: rl_logging_test <repository> <output-directory>");
        const auto repository = std::filesystem::canonical(argv[1]);
        const auto output = std::filesystem::absolute(argv[2]);
        const auto policy = output / "policy";
        std::filesystem::create_directories(policy / "exported");
        std::filesystem::create_directories(policy / "params");
        auto config = YAML::LoadFile((repository / "deploy/robots/g1/config/config.yaml").string());
        const auto source = (repository / "deploy/robots/g1" / config["FSM"]["Velocity_X"]["policy_dir"].as<std::string>()).lexically_normal();
        std::filesystem::create_symlink(source / "exported/policy.onnx", policy / "exported/policy.onnx");
        std::filesystem::copy_file(source / "params/deploy.yaml", policy / "params/deploy.yaml");
        config["FSM"]["Velocity_X"]["policy_dir"] = policy.string();
        config["FSM"]["Velocity_X"]["logging"] = true;
        config["FSM"]["Velocity_X"]["logging_dt"] = 0.001;
        config["FSM"]["Velocity_X"]["transitions"] = YAML::Node(YAML::NodeType::Map);
        config["FSM"]["Velocity_X"]["enable_bad_orientation_check"] = false;
        param::config = config;
        param::proj_dir = repository / "deploy/robots/g1";
        FSMStringMap.insert({1, "Passive"});

        // A separate DDS domain on loopback isolates all test publishers from robots.
        unitree::robot::ChannelFactory::Instance()->Init(156, "lo");
        FSMState::lowcmd = std::make_unique<LowCmd_t>();
        FSMState::lowstate = std::make_shared<LowState_t>();
        const auto deploy = YAML::LoadFile((policy / "params/deploy.yaml").string());
        const auto default_q = deploy["default_joint_pos"].as<std::vector<float>>();
        {
            std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
            auto& state = FSMState::lowstate->msg_;
            state.imu_state().quaternion()[0] = 1;
            state.imu_state().accelerometer()[2] = 9.81f;
            state.tick() = 42;
            for (std::size_t i = 0; i < default_q.size(); ++i) state.motor_state()[i].q() = default_q[i];
        }
        std::filesystem::path csv_path;
        {
            State_RLBase state(7, "Velocity_X");
            state.enter();
            struct ExitGuard {
                State_RLBase& state;
                ~ExitGuard() { state.exit(); }
            } exit_guard{state};
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            state.run();
            for (const auto& entry : std::filesystem::recursive_directory_iterator(policy / "logs")) {
                if (entry.path().extension() == ".csv") csv_path = entry.path();
            }
            check(!csv_path.empty(), "CSV not created");
            check(csv_path.parent_path().parent_path() == policy / "logs", "recording is not in its own directory");
            check(csv_path.parent_path().filename() == csv_path.stem(), "recording directory does not match its timestamp");
            auto rows = read_csv(csv_path);
            check(rows.size() == 2, "first row is not flushed before logger destruction");
            const auto header = rows[0];
            auto column = [&](const std::string& key) {
                const auto it = std::find(header.begin(), header.end(), key);
                check(it != header.end(), "missing CSV column: " + key);
                return static_cast<std::size_t>(it - header.begin());
            };
            check(rows[1][column("torso_imu_valid")] == "0", "missing torso is reported valid");
            check(rows[1][column("lowcmd_valid")] == "0", "missing LowCmd is reported valid");
            check(rows[1][column("lowcmd_config_id")] == "-1", "missing LowCmd has a configuration");
            check(std::isnan(std::stof(rows[1][column("torso_imu_quat_0")])), "missing torso data is not NaN");
            check(std::find(header.begin(), header.end(), "obs_base_ang_vel_0") == header.end(), "duplicated observation terms retained");
            check(std::find(header.begin(), header.end(), "temp_0") == header.end(), "temperature repeated in every row");
            check(std::find(header.begin(), header.end(), "cmd_fixed_0") == header.end(), "disabled fixed commands retained");
            check(std::find(header.begin(), header.end(), "q_des_0") == header.end(), "pending target duplicated alongside LowCmd");
            column("obs_979");

            unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::IMUState_> imu("rt/secondary_imu");
            unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_> command("rt/lowcmd");
            imu.InitChannel();
            command.InitChannel();
            unitree_hg::msg::dds_::IMUState_ imu_message;
            imu_message.quaternion()[0] = 0.9238795f;
            imu_message.quaternion()[2] = 0.3826834f;
            imu_message.rpy()[1] = 0.7853982f;
            unitree_hg::msg::dds_::LowCmd_ command_message;
            command_message.mode_machine() = 5;
            for (std::size_t i = 0; i < 29; ++i) {
                command_message.motor_cmd()[i].q() = 0.5f + 0.01f * i;
                command_message.motor_cmd()[i].kp() = 100.0f + i;
                command_message.motor_cmd()[i].kd() = 2.0f;
            }
            auto publish = [&] {
                for (int i = 0; i < 20; ++i) {
                    imu.Write(imu_message);
                    command.Write(command_message);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            };
            publish();
            state.run();
            rows = read_csv(csv_path);
            check(rows.back()[column("torso_imu_valid")] == "1", "torso subscription did not receive data");
            check(rows.back()[column("lowcmd_valid")] == "1", "LowCmd subscription did not receive data");
            check(std::abs(std::stof(rows.back()[column("lowcmd_q_14")]) - 0.64f) < 0.0001f,
                  "LowCmd was read from the pending policy target instead of the published topic");
            check(rows.back()[column("lowcmd_config_id")] == "0", "first configuration ID incorrect");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            state.run();
            check(read_csv(csv_path).back()[column("lowcmd_config_id")] == "0", "unchanged gains created another configuration");

            command_message.motor_cmd()[14].kp() = 99.5f;
            command_message.motor_cmd()[14].kd() = 3.5f;
            command_message.motor_cmd()[14].tau() = 0.25f;
            publish();
            {
                std::lock_guard<std::mutex> lock(FSMState::lowstate->mutex_);
                FSMState::lowstate->msg_.motor_state()[0].temperature()[0] = 45;
            }
            state.run();
            rows = read_csv(csv_path);
            check(rows.back()[column("lowcmd_config_id")] == "1", "changed gains were not recorded");
            check(rows.back()[column("temperature_id")] == "1", "temperature change was lost");
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            state.run();
            rows = read_csv(csv_path);
            check(rows.back()[column("torso_imu_valid")] == "0", "stale torso data reported fresh");
            check(rows.back()[column("lowcmd_valid")] == "0", "stale LowCmd reported fresh");
            for (const auto& row : rows) check(row.size() == header.size(), "incomplete CSV row");
            state.exit();
            std::cout << "CSV_COLUMNS=" << header.size() << " ROWS=" << rows.size() - 1 << std::endl;
        }
        auto metadata_path = csv_path;
        metadata_path.replace_extension(".metadata.json");
        std::ifstream metadata_file(metadata_path);
        nlohmann::json metadata;
        metadata_file >> metadata;
        check(metadata["format_version"] == 2, "metadata format missing");
        check(metadata["observation_terms"].size() == 7, "observation layout missing");
        auto events_path = csv_path;
        events_path.replace_extension(".events.jsonl");
        std::ifstream events_file(events_path);
        int gain_events = 0, temperature_events = 0;
        for (std::string line; std::getline(events_file, line);) {
            const auto event = nlohmann::json::parse(line);
            if (event["kind"] == "lowcmd") {
                ++gain_events;
                if (event["id"] == 1) {
                    check(event["values"]["kp"][14] == 99.5, "actual kp change missing");
                    check(event["values"]["kd"][14] == 3.5, "actual kd change missing");
                    check(event["values"]["tau"][14] == 0.25, "actual feedforward torque missing");
                }
            } else if (event["kind"] == "temperature") ++temperature_events;
        }
        check(gain_events == 2, "constant gains were repeatedly recorded");
        check(temperature_events == 2, "constant temperature was repeatedly recorded");
        FSMState::lowstate.reset();
        FSMState::lowcmd.reset();
        std::cout << "PASS published telemetry, gain/temperature changes, stale/missing data, compact schema, and live CSV flush" << std::endl;
        std::cout << csv_path << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
