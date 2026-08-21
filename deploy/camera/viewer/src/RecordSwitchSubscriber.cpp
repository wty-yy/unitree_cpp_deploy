#include "RecordSwitchSubscriber.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

namespace camera_viewer
{

RecordSwitchSubscriber::RecordSwitchSubscriber(const DdsConfig& config)
    : subscriber_(config.control_topic), queue_length_(config.queue_length)
{
}

RecordSwitchSubscriber::~RecordSwitchSubscriber()
{
    Stop();
}

void RecordSwitchSubscriber::Start()
{
    if (started_) {
        return;
    }
    subscriber_.InitChannel(
        [this](const void* message) { OnMessage(message); }, queue_length_);
    started_ = true;
}

void RecordSwitchSubscriber::Stop()
{
    if (!started_) {
        return;
    }
    subscriber_.CloseChannel();
    started_ = false;
}

std::optional<bool> RecordSwitchSubscriber::ConsumeCommand()
{
    const int state = requested_state_.exchange(-1);
    if (state < 0) {
        return std::nullopt;
    }
    return state != 0;
}

void RecordSwitchSubscriber::OnMessage(const void* raw_message)
{
    if (raw_message == nullptr) {
        if (!invalid_warning_emitted_.exchange(true)) {
            std::cerr << "[WARN] Ignoring null record-switch DDS sample" << std::endl;
        }
        return;
    }

    std::string command = static_cast<const Message*>(raw_message)->data();
    command.erase(
        std::remove_if(command.begin(), command.end(), [](unsigned char character) {
            return std::isspace(character) != 0;
        }),
        command.end());
    std::transform(command.begin(), command.end(), command.begin(), [](unsigned char character) {
        return static_cast<char>(std::toupper(character));
    });

    if (command == "ON") {
        requested_state_.store(1);
        invalid_warning_emitted_.store(false);
    } else if (command == "OFF") {
        requested_state_.store(0);
        invalid_warning_emitted_.store(false);
    } else if (!invalid_warning_emitted_.exchange(true)) {
        std::cerr << "[WARN] Ignoring record-switch command '" << command
                  << "'; expected ON or OFF" << std::endl;
    }
}

}  // namespace camera_viewer
