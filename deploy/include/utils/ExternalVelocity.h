#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace robot_automation {

// A local, acknowledged velocity lease. The FSM and policy threads never do socket I/O.
class ExternalVelocity {
public:
    using Json = nlohmann::json;
    using Ranges = std::array<std::array<double, 2>, 3>;
    static bool accepts_external_velocity(const std::string& state) {
        return state == "Velocity_X" || state == "Velocity_A";
    }
    struct MotorHealth {
        uint32_t motorstate = 0;
        std::array<int16_t, 2> temperature_c{};
    };
    using MotorHealthSample = std::array<MotorHealth, 29>;
    static ExternalVelocity& instance() {
        static ExternalVelocity bridge;
        return bridge;
    }

    static double now() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void start(const std::string& path = "") {
        if (running_) throw std::runtime_error("Velocity bridge already started");
        const std::string directory = "/tmp/robot-automation-" + std::to_string(geteuid());
        if (path.empty()) {
            if (mkdir(directory.c_str(), 0700) != 0 && errno != EEXIST)
                throw std::runtime_error("Cannot create velocity socket directory");
            struct stat info{};
            if (lstat(directory.c_str(), &info) != 0 || !S_ISDIR(info.st_mode) ||
                info.st_uid != geteuid() || (info.st_mode & 0077) != 0)
                throw std::runtime_error("Velocity socket directory must be private and owned by this user");
        }
        path_ = path.empty() ? directory + "/velocity.sock" : path;
        if (path_.size() >= sizeof(sockaddr_un::sun_path))
            throw std::runtime_error("Velocity socket path is too long");
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::strcpy(address.sun_path, path_.c_str());
        struct stat info{};
        if (lstat(path_.c_str(), &info) == 0) {
            if (!S_ISSOCK(info.st_mode) || info.st_uid != geteuid())
                throw std::runtime_error("Refusing to replace non-owned velocity socket");
            int probe = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
            const int result = connect(probe, reinterpret_cast<sockaddr*>(&address), sizeof(address));
            const int error = errno;
            close(probe);
            if (result == 0 || error != ECONNREFUSED)
                throw std::runtime_error("Velocity socket is already in use");
            if (unlink(path_.c_str()) != 0)
                throw std::runtime_error("Cannot remove stale velocity socket");
        }
        fd_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        if (fd_ < 0 || bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
            if (fd_ >= 0) close(fd_);
            fd_ = -1;
            throw std::runtime_error("Cannot bind velocity socket");
        }
        if (chmod(path_.c_str(), 0600) != 0) {
            close(fd_); fd_ = -1; unlink(path_.c_str());
            throw std::runtime_error("Cannot protect velocity socket");
        }
        running_ = true;
        worker_ = std::thread([this] { serve(); });
    }

    ~ExternalVelocity() {
        running_ = false;
        if (worker_.joinable()) worker_.join();
        if (fd_ >= 0) { close(fd_); unlink(path_.c_str()); }
    }

    void update(const std::string& state, bool manual, bool robot_connected) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (state != state_) {
            clear("FSM changed to " + state);
            ranges_ = {};
            limits_ready_ = false;
            boost_active_ = false;
            state_ = state;
            ++generation_;
        }
        manual_ = manual;
        connected_ = robot_connected;
        updated_ = now();
        if (manual) clear("Manual controller takeover");
        else if (!robot_connected) clear("Robot state timed out");
        else if (!accepts_external_velocity(state)) clear("Only Velocity_X or Velocity_A accepts external velocity");
    }

    // Called by the policy observation with the same ranges used for its joystick.
    bool set_ranges(const Ranges& ranges, bool boost_active) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!accepts_external_velocity(state_)) return false;
        for (const auto& axis : ranges) {
            if (!std::isfinite(axis[0]) || !std::isfinite(axis[1]) ||
                axis[0] > 0 || axis[1] < 0 || axis[0] > axis[1]) {
                invalidate_ranges_locked("Invalid policy velocity ranges");
                return false;
            }
        }
        ranges_ = ranges;
        boost_active_ = boost_active;
        limits_ready_ = true;
        if (armed_ && now() < deadline_ && !within_ranges(velocity_))
            clear("Policy velocity ranges changed; active command is outside the new limits");
        return true;
    }

    void invalidate_ranges() {
        std::lock_guard<std::mutex> guard(mutex_);
        invalidate_ranges_locked("Policy velocity ranges unavailable");
    }

    // Queue one FSM transition request; the FSM thread consumes it and reports back.
    bool take_requested_state(std::string& target) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!state_request_pending_) return false;
        target = requested_state_;
        state_request_pending_ = false;
        return true;
    }

    void report_state_result(bool accepted, const std::string& message) {
        std::lock_guard<std::mutex> guard(mutex_);
        state_request_ok_ = accepted;
        state_request_message_ = message;
    }

    // Current AI-held buttons/axes; false once the TTL expires (auto-release).
    bool virtual_joystick(std::vector<std::string>& buttons, bool& has_axes,
                          std::array<float, 4>& axes) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!virtual_active_) return false;
        if (now() - virtual_updated_ > virtual_ttl_s_) return false;
        buttons = virtual_buttons_;
        has_axes = virtual_has_axes_;
        axes = virtual_axes_;
        return true;
    }

    void update_motor_health(uint32_t tick, const MotorHealthSample& sample) {
        std::lock_guard<std::mutex> guard(mutex_);
        // Re-reading the same DDS sample does not make its telemetry fresh.
        if (motor_health_available_ && tick == motor_health_tick_) return;
        motor_health_ = sample;
        motor_health_tick_ = tick;
        motor_health_updated_ = now();
        motor_health_available_ = true;
    }

    bool sample(std::array<float, 3>& command) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!ready()) {
            if (armed_) clear("FSM unavailable or manual takeover");
            return false;
        }
        if (!armed_) return false;
        if (now() >= deadline_) velocity_ = {0, 0, 0};
        for (std::size_t i = 0; i < command.size(); ++i)
            command[i] = static_cast<float>(velocity_[i]);
        applied_ = command;
        return true;
    }

    Json handle(const Json& request) {
        std::lock_guard<std::mutex> guard(mutex_);
        Json response = handle_locked(request);
        response["owns_lease"] = armed_ && request.is_object() &&
            request.contains("session") && request["session"].is_string() &&
            request["session"].get<std::string>() == owner_;
        if (request.is_object() && request.contains("request_id"))
            response["request_id"] = request["request_id"];
        return response;
    }

private:
    Json handle_locked(const Json& request) {
        try {
            const std::string type = request.at("type").get<std::string>();
            if (type == "status") return status(true, "");
            if (type == "disarm") {
                clear("External control released");
                return status(true, "");
            }
            if (type == "state") {
                const std::string target = request.at("target").get<std::string>();
                if (target.empty() || target.size() > 64)
                    return status(false, "Invalid FSM state name");
                requested_state_ = target;
                state_request_pending_ = true;
                state_request_ok_ = true;
                state_request_message_ = "FSM state request queued: " + target;
                reason_ = state_request_message_;
                return status(true, "");
            }
            if (type == "joystick") {
                std::vector<std::string> buttons;
                if (request.contains("buttons") && request["buttons"].is_array()) {
                    for (const auto& item : request["buttons"])
                        buttons.push_back(item.get<std::string>());
                }
                bool has_axes = false;
                std::array<float, 4> axes{0.0f, 0.0f, 0.0f, 0.0f};
                if (request.contains("axes") && request["axes"].is_object()) {
                    const auto& node = request["axes"];
                    has_axes = true;
                    axes = {static_cast<float>(node.value("lx", 0.0)),
                            static_cast<float>(node.value("ly", 0.0)),
                            static_cast<float>(node.value("rx", 0.0)),
                            static_cast<float>(node.value("ry", 0.0))};
                }
                const int ttl = request.value("ttl_ms", 1000);
                if (ttl < 50 || ttl > 30000)
                    return status(false, "Joystick TTL must be 50..30000 ms");
                virtual_buttons_ = buttons;
                virtual_has_axes_ = has_axes;
                virtual_axes_ = axes;
                virtual_updated_ = now();
                virtual_ttl_s_ = ttl / 1000.0;
                virtual_active_ = true;
                reason_ = "Virtual joystick input updated";
                return status(true, "");
            }
            if (type != "arm" && type != "velocity" && type != "stop")
                return status(false, "Unknown request type");
            const std::string session = request.at("session").get<std::string>();
            if (session.size() < 16 || session.size() > 128)
                return status(false, "Invalid session");
            const double sent = request.at("sent_mono_s").get<double>();
            const double age = now() - sent;
            if (!std::isfinite(sent) || age < -0.05 || age > 0.15)
                return status(false, "Expired request timestamp");
            if (type == "arm") {
                if (!limits_ready_) return status(false, "Policy velocity ranges unavailable");
                if (!ready()) return status(false, "Enter Velocity_X or Velocity_A and release the handheld controls first");
                if (armed_ && session != owner_ && now() < deadline_)
                    return status(false, "Another active client owns the velocity lease");
                // An acknowledgement retry must not reset sequence checking or the lease.
                if (armed_ && session == owner_) return status(true, "");
                owner_ = session;
                armed_ = true;
                velocity_ = {0, 0, 0};
                applied_ = {0, 0, 0};
                deadline_ = 0;
                sequence_ = 0;
                reason_ = "External control armed; waiting for velocity";
                return status(true, "");
            }
            if (type == "stop") {
                if (armed_ && session != owner_) return status(false, "Not the lease owner");
                velocity_ = {0, 0, 0};
                applied_ = {0, 0, 0};
                deadline_ = 0;
                reason_ = "External velocity stopped";
                return status(true, "");
            }
            if (!ready() || !armed_ || owner_ != session)
                return status(false, "External control is not armed for this session");
            const auto& sequence_json = request.at("sequence");
            if (!sequence_json.is_number_integer() ||
                (!sequence_json.is_number_unsigned() && sequence_json.get<int64_t>() <= 0))
                return status(false, "Sequence must be a positive integer");
            const auto sequence = sequence_json.get<uint64_t>();
            if (sequence <= sequence_) return status(false, "Out-of-order velocity command");
            const double vx = request.at("vx").get<double>();
            const double vy = request.at("vy").get<double>();
            const double wz = request.at("yaw_rate").get<double>();
            const auto& ttl_json = request.at("ttl_ms");
            if (!ttl_json.is_number_integer() || ttl_json < 50 || ttl_json > 300)
                return status(false, "TTL must be an integer from 50 through 300 milliseconds");
            const int ttl = ttl_json.get<int>();
            if (!within_ranges({vx, vy, wz}))
                return status(false, "Velocity exceeds current policy ranges");
            // Use sender time so a delayed packet cannot renew an obsolete lease.
            const double deadline = sent + ttl / 1000.0;
            if (deadline <= now()) return status(false, "Velocity lease already expired");
            velocity_ = {vx, vy, wz};
            deadline_ = deadline;
            sequence_ = sequence;
            reason_ = "External velocity accepted";
            return status(true, "");
        } catch (const std::exception& exc) {
            return status(false, std::string("Invalid request: ") + exc.what());
        }
    }

    bool ready() const {
        return accepts_external_velocity(state_) && connected_ && !manual_ && limits_ready_ && now() - updated_ < 0.1;
    }

    bool within_ranges(const std::array<double, 3>& velocity) const {
        if (!limits_ready_) return false;
        for (std::size_t i = 0; i < velocity.size(); ++i) {
            if (!std::isfinite(velocity[i]) || velocity[i] < ranges_[i][0] || velocity[i] > ranges_[i][1])
                return false;
        }
        return true;
    }

    void invalidate_ranges_locked(const std::string& reason) {
        clear(reason);
        ranges_ = {};
        limits_ready_ = false;
        boost_active_ = false;
    }

    void clear(const std::string& reason) {
        armed_ = false;
        owner_.clear();
        velocity_ = {0, 0, 0};
        applied_ = {0, 0, 0};
        deadline_ = 0;
        reason_ = reason;
    }

    Json status(bool ok, const std::string& error) {
        if (armed_ && !ready()) clear("FSM heartbeat lost or control unavailable");
        const bool active = armed_ && now() < deadline_;
        if (!active) velocity_ = {0, 0, 0};
        return {{"ok", ok}, {"error", error}, {"state", state_},
                {"connected", connected_ && now() - updated_ < 0.1},
                {"manual_override", manual_}, {"armed", armed_},
                {"remote_active", active}, {"ready_to_arm", ready()},
                {"generation", generation_}, {"reason", reason_},
                {"requested_state", state_request_pending_ ? requested_state_ : ""},
                {"state_request_ok", state_request_ok_},
                {"state_request_message", state_request_message_},
                {"virtual_joystick_active",
                    virtual_active_ && now() - virtual_updated_ <= virtual_ttl_s_},
                {"virtual_buttons", virtual_buttons_},
                {"velocity_ranges", {{"vx", ranges_[0]}, {"vy", ranges_[1]}, {"yaw_rate", ranges_[2]}}},
                {"boost_active", boost_active_}, {"limits_ready", limits_ready_},
                {"motor_health", motor_health_status()},
                {"velocity", velocity_}, {"applied_velocity", applied_},
                {"lease_remaining_s", std::max(0.0, deadline_ - now())}};
    }

    Json motor_health_status() const {
        Json result = {{"available", motor_health_available_}, {"fresh", false},
                       {"age_s", nullptr}, {"sample_tick", nullptr},
                       {"error_count", nullptr}, {"motors", Json::array()}};
        if (!motor_health_available_) return result;
        const double age = std::max(0.0, now() - motor_health_updated_);
        result["fresh"] = connected_ && now() - updated_ < 0.1 && age <= 0.5;
        result["age_s"] = age;
        result["sample_tick"] = motor_health_tick_;
        int error_count = 0;
        for (std::size_t i = 0; i < motor_health_.size(); ++i) {
            const auto& motor = motor_health_[i];
            if (motor.motorstate != 0) ++error_count;
            result["motors"].push_back({{"id", i}, {"motorstate", motor.motorstate},
                                         {"temperature_c", motor.temperature_c}});
        }
        result["error_count"] = error_count;
        return result;
    }

    void serve() {
        while (running_) {
            pollfd descriptor{fd_, POLLIN, 0};
            if (poll(&descriptor, 1, 50) <= 0) continue;
            char data[2048];
            sockaddr_un peer{};
            socklen_t peer_length = sizeof(peer);
            const ssize_t size = recvfrom(fd_, data, sizeof(data), 0,
                                         reinterpret_cast<sockaddr*>(&peer), &peer_length);
            if (size <= 0) continue;
            const auto request = Json::parse(data, data + size, nullptr, false);
            Json response = handle(request);
            const std::string payload = response.dump();
            sendto(fd_, payload.data(), payload.size(), MSG_DONTWAIT | MSG_NOSIGNAL,
                   reinterpret_cast<sockaddr*>(&peer), peer_length);
        }
    }

    std::mutex mutex_;
    std::string state_ = "Initializing", owner_, reason_ = "Not armed", path_;
    std::string requested_state_, state_request_message_;
    bool state_request_pending_ = false, state_request_ok_ = true;
    std::vector<std::string> virtual_buttons_;
    bool virtual_has_axes_ = false, virtual_active_ = false;
    std::array<float, 4> virtual_axes_{0.0f, 0.0f, 0.0f, 0.0f};
    double virtual_updated_ = 0, virtual_ttl_s_ = 1.0;
    bool manual_ = false, connected_ = false, armed_ = false;
    bool boost_active_ = false, limits_ready_ = false;
    Ranges ranges_{};
    MotorHealthSample motor_health_{};
    bool motor_health_available_ = false;
    uint32_t motor_health_tick_ = 0;
    double motor_health_updated_ = 0;
    double updated_ = 0, deadline_ = 0;
    uint64_t sequence_ = 0, generation_ = 0;
    std::array<double, 3> velocity_{0, 0, 0};
    std::array<float, 3> applied_{0, 0, 0};
    int fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread worker_;
};

} // namespace robot_automation
