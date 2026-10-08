#include "utils/ExternalVelocity.h"
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

using Bridge = robot_automation::ExternalVelocity;
using Json = Bridge::Json;

namespace {
const std::string owner(32, 'a');
const std::string other(32, 'b');
const Bridge::Ranges base_ranges{{{-2.0, 2.0}, {-1.0, 1.0}, {-2.0, 2.0}}};
const Bridge::Ranges boost_ranges{{{-3.0, 3.0}, {-0.5, 0.5}, {-1.0, 1.0}}};
const Bridge::Ranges asymmetric_ranges{{{-0.7, 1.4}, {-0.25, 0.45}, {-0.8, 0.6}}};
int assertions = 0;

void check(bool condition, const char* message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}

Json request(const std::string& type, const std::string& session = owner) {
    return {{"type", type}, {"session", session}, {"sent_mono_s", Bridge::now()},
            {"request_id", "cpp-test"}};
}

Json velocity(uint64_t sequence = 1, int ttl = 100) {
    auto message = request("velocity");
    message.update({{"sequence", sequence}, {"ttl_ms", ttl},
                    {"vx", 0.2}, {"vy", -0.1}, {"yaw_rate", 0.3}});
    return message;
}

void fresh_arm(Bridge& bridge, const Bridge::Ranges& ranges = base_ranges, bool boost = false) {
    bridge.update("Passive", false, true);
    bridge.update("Velocity_X", false, true);
    check(bridge.set_ranges(ranges, boost), "valid policy ranges rejected");
    check(bridge.handle(request("arm"))["ok"], "fresh arm rejected");
}

void run_motor_health_tests(Bridge& bridge) {
    auto health = bridge.handle(request("status"))["motor_health"];
    check(!health["available"].get<bool>() && !health["fresh"].get<bool>(), "missing motor telemetry reported available");
    check(health["age_s"].is_null() && health["sample_tick"].is_null() &&
          health["error_count"].is_null() && health["motors"].empty(), "missing telemetry reported healthy");
    bridge.update("Passive", false, true);
    Bridge::MotorHealthSample sample{};
    sample[11].motorstate = 512;
    sample[11].temperature_c = {44, 91};
    bridge.update_motor_health(100, sample);
    health = bridge.handle(request("status"))["motor_health"];
    check(health["available"] && health["fresh"] && health["sample_tick"] == 100, "fresh telemetry not reported");
    check(health["error_count"] == 1 && health["motors"].size() == 29, "motor count or error count incorrect");
    check(health["motors"][11]["id"] == 11 && health["motors"][11]["motorstate"] == 512 &&
          health["motors"][11]["temperature_c"] == Json({44, 91}), "motor raw data changed");
    sample[11].motorstate = 0;
    bridge.update_motor_health(100, sample);
    check(bridge.handle(request("status"))["motor_health"]["error_count"] == 1, "duplicate tick replaced sample");
    bridge.update_motor_health(101, sample);
    check(bridge.handle(request("status"))["motor_health"]["error_count"] == 0, "cleared motor fault was retained");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    bridge.update_motor_health(101, sample);
    std::this_thread::sleep_for(std::chrono::milliseconds(220));
    bridge.update("Passive", false, true);
    health = bridge.handle(request("status"))["motor_health"];
    check(health["available"] && !health["fresh"].get<bool>() && health["age_s"].get<double>() >= 0.5,
          "duplicate tick falsely refreshed telemetry");
    check(health["motors"].size() == 29 && health["error_count"] == 0, "stale telemetry did not retain last values");
    bridge.update_motor_health(102, sample);
    bridge.update("Passive", false, false);
    check(!bridge.handle(request("status"))["motor_health"]["fresh"].get<bool>(), "disconnected telemetry reported fresh");
    bridge.update("Passive", false, true);
    bridge.update_motor_health(103, sample);
    check(bridge.handle(request("status"))["motor_health"]["fresh"], "recovered telemetry not fresh");
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    check(!bridge.handle(request("status"))["motor_health"]["fresh"].get<bool>(), "lost FSM heartbeat reported fresh telemetry");
    bridge.update("Passive", false, true);
    bridge.update_motor_health(0, sample);
    check(bridge.handle(request("status"))["motor_health"]["sample_tick"] == 0, "tick reset not accepted");
}

void run_tests(Bridge& bridge) {
    run_motor_health_tests(bridge);
    std::array<float, 3> output{};
    bridge.update("Passive", false, true);
    check(!bridge.handle(request("arm"))["ok"].get<bool>(), "Passive must reject arm");
    check(!bridge.set_ranges(base_ranges, false), "Passive accepted policy ranges");
    bridge.update("Velocity_X", false, true);
    check(!bridge.handle(request("status"))["limits_ready"].get<bool>(), "missing limits reported ready");
    check(!bridge.handle(request("arm"))["ok"].get<bool>(), "missing limits accepted arm");
    check(!bridge.handle(velocity())["ok"].get<bool>(), "missing limits accepted velocity");
    fresh_arm(bridge);
    auto status = bridge.handle(request("status"));
    check(status["owns_lease"], "owner must be recognized");
    check(status["request_id"] == "cpp-test", "request ID not echoed");
    check(!bridge.handle(request("status", other))["owns_lease"].get<bool>(), "other client claimed ownership");
    check(bridge.sample(output) && output == std::array<float, 3>{0, 0, 0}, "arming must hold zero");
    check(bridge.handle(velocity())["ok"], "valid velocity rejected");
    check(bridge.sample(output) && std::abs(output[0] - 0.2f) < 0.001f, "velocity not sampled");
    check(!bridge.handle(request("arm", other))["ok"].get<bool>(), "active owner was replaced");
    check(!bridge.handle(velocity())["ok"].get<bool>(), "duplicate sequence accepted");
    check(bridge.handle(request("arm"))["ok"], "arm retry failed");
    check(!bridge.handle(velocity())["ok"].get<bool>(), "arm retry reset sequence");
    check(!bridge.handle(request("stop", other))["ok"].get<bool>(), "non-owner stop accepted");

    const std::array<const char*, 3> fields{"vx", "vy", "yaw_rate"};
    for (const auto& ranges : {base_ranges, boost_ranges, asymmetric_ranges}) {
        const bool boost = ranges == boost_ranges;
        for (std::size_t axis = 0; axis < fields.size(); ++axis) {
            for (std::size_t side = 0; side < 2; ++side) {
                fresh_arm(bridge, ranges, boost);
                auto boundary = velocity();
                boundary[fields[axis]] = ranges[axis][side];
                check(bridge.handle(boundary)["ok"], "velocity at policy limit rejected");
                check(bridge.sample(output) &&
                      std::abs(output[axis] - ranges[axis][side]) < 0.00001,
                      "velocity at policy limit not sampled exactly");
                bridge.set_ranges(ranges, boost);
                auto reported = bridge.handle(request("status"));
                check(reported["armed"], "identical range update revoked a valid boundary command");
                check(reported["velocity_ranges"][fields[axis]] == Json(ranges[axis]),
                      "reported range differs from selected policy range");
                check(reported["boost_active"] == boost, "incorrect boost flag");
                auto outside = velocity(2);
                outside[fields[axis]] = ranges[axis][side] + (side == 0 ? -0.0001 : 0.0001);
                check(!bridge.handle(outside)["ok"].get<bool>(), "velocity beyond policy limit accepted");
            }
        }
    }

    fresh_arm(bridge);
    auto lateral = velocity();
    lateral["vy"] = 0.8;
    check(bridge.handle(lateral)["ok"], "base lateral command rejected");
    bridge.set_ranges(boost_ranges, true);
    auto changed = bridge.handle(request("status"));
    check(!changed["armed"].get<bool>() && changed["velocity"] == Json({0, 0, 0}),
          "boost range narrowing did not revoke lateral command");
    check(changed["limits_ready"], "valid range switch lost readiness");
    fresh_arm(bridge, boost_ranges, true);
    auto fast = velocity();
    fast["vx"] = 2.5;
    check(bridge.handle(fast)["ok"], "boost longitudinal command rejected");
    bridge.set_ranges(base_ranges, false);
    changed = bridge.handle(request("status"));
    check(!changed["armed"].get<bool>() && changed["velocity"] == Json({0, 0, 0}),
          "leaving boost did not revoke excessive longitudinal command");
    fresh_arm(bridge);
    check(bridge.handle(velocity())["ok"], "in-range switch command rejected");
    bridge.set_ranges(boost_ranges, true);
    check(bridge.handle(request("status"))["armed"], "valid command unnecessarily revoked on boost");
    bridge.set_ranges(base_ranges, false);
    check(bridge.handle(request("status"))["armed"], "valid command unnecessarily revoked on base");
    bridge.invalidate_ranges();
    changed = bridge.handle(request("status"));
    check(!changed["armed"].get<bool>() && !changed["limits_ready"].get<bool>(),
          "missing ranges did not revoke control");
    check(!bridge.handle(request("arm"))["ok"].get<bool>(), "missing ranges allowed rearm");
    check(!bridge.handle(velocity(2))["ok"].get<bool>(), "missing ranges allowed velocity");
    for (const auto& invalid : {
             Bridge::Ranges{{{1.0, 2.0}, {-1.0, 1.0}, {-2.0, 2.0}}},
             Bridge::Ranges{{{-2.0, -1.0}, {-1.0, 1.0}, {-2.0, 2.0}}},
             Bridge::Ranges{{{2.0, -2.0}, {-1.0, 1.0}, {-2.0, 2.0}}},
             Bridge::Ranges{{{-2.0, std::numeric_limits<double>::infinity()}, {-1.0, 1.0}, {-2.0, 2.0}}},
             Bridge::Ranges{{{-2.0, std::numeric_limits<double>::quiet_NaN()}, {-1.0, 1.0}, {-2.0, 2.0}}}}) {
        check(!bridge.set_ranges(invalid, false), "invalid range accepted");
        check(!bridge.handle(request("arm"))["ok"].get<bool>(), "invalid range allowed arm");
    }
    fresh_arm(bridge);
    check(bridge.handle(velocity())["ok"], "velocity after boundary tests rejected");
    for (const auto& bad_sequence : std::vector<Json>{-1, 0, 1.5, true, "2"}) {
        auto bad = velocity(2);
        bad["sequence"] = bad_sequence;
        check(!bridge.handle(bad)["ok"].get<bool>(), "invalid sequence accepted");
    }
    for (const auto& bad_ttl : std::vector<Json>{49, 301, 99.5, true, "100", 4294967396ULL}) {
        auto bad = velocity(2);
        bad["ttl_ms"] = bad_ttl;
        check(!bridge.handle(bad)["ok"].get<bool>(), "invalid TTL accepted");
    }
    auto stale = velocity(2);
    stale["sent_mono_s"] = Bridge::now() - 1;
    check(!bridge.handle(stale)["ok"].get<bool>(), "stale command accepted");
    stale["sent_mono_s"] = Bridge::now() + 1;
    check(!bridge.handle(stale)["ok"].get<bool>(), "future command accepted");
    stale["sent_mono_s"] = Bridge::now() - 0.08;
    stale["ttl_ms"] = 50;
    check(!bridge.handle(stale)["ok"].get<bool>(), "expired lease accepted");
    check(bridge.handle(request("stop"))["ok"], "stop rejected");
    check(bridge.sample(output) && output[0] == 0, "stop must sample zero");

    check(bridge.handle(velocity(2, 50))["ok"], "short lease rejected");
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    bridge.update("Velocity_X", false, true);
    check(bridge.sample(output) && output == std::array<float, 3>{0, 0, 0}, "expired lease not zeroed");
    check(bridge.handle(request("status"))["armed"], "TTL expiry need not disarm");

    bridge.update("Velocity_X", true, true);
    check(!bridge.sample(output), "manual takeover not released to joystick");
    check(!bridge.handle(request("arm"))["ok"].get<bool>(), "manual control accepted arm");
    bridge.update("Velocity_X", false, true);
    check(!bridge.handle(velocity(3))["ok"].get<bool>(), "manual takeover did not require rearm");
    fresh_arm(bridge);
    check(bridge.handle(velocity())["ok"], "velocity after rearm rejected");
    bridge.update("Velocity_Y", false, true);
    check(!bridge.sample(output), "wrong policy sampled external command");
    check(!bridge.handle(request("status"))["limits_ready"].get<bool>(), "FSM exit kept range readiness");
    bridge.update("Velocity_X", false, true);
    check(!bridge.handle(velocity(2))["ok"].get<bool>(), "state change did not require rearm");
    fresh_arm(bridge);
    bridge.update("Velocity_X", false, false);
    check(!bridge.handle(request("status"))["armed"].get<bool>(), "disconnect did not disarm");
    fresh_arm(bridge);
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    check(!bridge.sample(output), "stale FSM heartbeat still sampled external command");
    check(!bridge.handle(request("status"))["connected"].get<bool>(), "stale heartbeat reported connected");
    check(!bridge.handle(Json::parse("invalid", nullptr, false))["ok"].get<bool>(), "malformed JSON accepted");
    check(!bridge.handle(Json{{"type", "velocity"}})["ok"].get<bool>(), "missing fields accepted");
    check(bridge.handle(request("disarm"))["ok"], "disarm rejected");

    std::atomic<bool> finish{false};
    std::thread fsm([&] {
        while (!finish) {
            bridge.update("Velocity_X", false, true);
            bridge.set_ranges(base_ranges, false);
            bridge.sample(output);
            bridge.update("Passive", false, true);
        }
    });
    for (int i = 0; i < 1000; ++i) {
        bridge.handle(request("arm"));
        bridge.handle(velocity(i + 1));
        bridge.handle(request("status"));
    }
    finish = true;
    fsm.join();
    std::cout << "PASS " << assertions << " assertions and 1000 concurrent request cycles\n";
}

void serve_test(Bridge& bridge, const std::string& path) {
    std::atomic<bool> done{false};
    std::mutex state_mutex;
    std::string state = "Passive";
    bool manual = false, connected = true, heartbeat = true;
    bool boost = false, ranges_available = true;
    auto ranges = base_ranges;
    bool health_updates = true;
    uint32_t sample_tick = 0;
    Bridge::MotorHealthSample motors{};
    for (auto& motor : motors) motor.temperature_c = {30, 35};
    bridge.update(state, manual, connected);
    bridge.start(path);
    std::thread fsm([&] {
        while (!done) {
            {
                std::lock_guard<std::mutex> guard(state_mutex);
                if (heartbeat) bridge.update(state, manual, connected);
                if (heartbeat && health_updates) bridge.update_motor_health(++sample_tick, motors);
                if (heartbeat && state == "Velocity_X") {
                    if (ranges_available) bridge.set_ranges(ranges, boost);
                    else bridge.invalidate_ranges();
                }
            }
            std::array<float, 3> output{};
            bridge.sample(output);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    std::cout << "READY " << path << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit") break;
        std::lock_guard<std::mutex> guard(state_mutex);
        if (line == "manual") manual = true;
        else if (line == "neutral") manual = false;
        else if (line == "disconnect") connected = false;
        else if (line == "connect") connected = true;
        else if (line == "stale") heartbeat = false;
        else if (line == "fresh") heartbeat = true;
        else if (line == "base") { ranges = base_ranges; boost = false; ranges_available = true; }
        else if (line == "boost") { ranges = boost_ranges; boost = true; ranges_available = true; }
        else if (line == "asymmetric") { ranges = asymmetric_ranges; boost = false; ranges_available = true; }
        else if (line == "nolimits") ranges_available = false;
        else if (line == "motor_fault") { motors[11].motorstate = 512; motors[11].temperature_c = {44, 91}; }
        else if (line == "motor_clear") { motors[11].motorstate = 0; motors[11].temperature_c = {35, 40}; }
        else if (line == "motor_stale") health_updates = false;
        else if (line == "motor_fresh") health_updates = true;
        else if (line == "Velocity_X" || line == "Passive" || line == "Velocity_Y") state = line;
        else { std::cout << "ERROR unknown simulation command" << std::endl; continue; }
        if (heartbeat) bridge.update(state, manual, connected);
        if (heartbeat && health_updates) bridge.update_motor_health(++sample_tick, motors);
        if (heartbeat && state == "Velocity_X") {
            if (ranges_available) bridge.set_ranges(ranges, boost);
            else bridge.invalidate_ranges();
        }
        std::cout << "STATE " << bridge.handle(request("status")).dump() << std::endl;
    }
    done = true;
    fsm.join();
}
} // namespace

int main(int argc, char** argv) {
    try {
        auto& bridge = Bridge::instance();
        if (argc == 3 && std::string(argv[1]) == "--serve-test") serve_test(bridge, argv[2]);
        else if (argc == 1) run_tests(bridge);
        else throw std::runtime_error("Usage: external_velocity_test [--serve-test SOCKET]");
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << std::endl;
        return 1;
    }
}
