#include "WirelessControllerSubscriber.h"

#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

namespace camera_viewer
{

WirelessControllerSubscriber::WirelessControllerSubscriber(const DdsConfig& config)
    : subscriber_(config.wireless_controller_topic),
      queue_length_(config.queue_length)
{
}

WirelessControllerSubscriber::~WirelessControllerSubscriber()
{
    Stop();
}

void WirelessControllerSubscriber::Start()
{
    if (started_) {
        return;
    }
    subscriber_.InitChannel(
        [this](const void* message) { OnMessage(message); }, queue_length_);
    started_ = true;
}

void WirelessControllerSubscriber::Stop()
{
    if (!started_) {
        return;
    }
    subscriber_.CloseChannel();
    started_ = false;
}

unsigned int WirelessControllerSubscriber::ConsumeToggleCount()
{
    return pending_toggles_.exchange(0);
}

void WirelessControllerSubscriber::OnMessage(const void* raw_message)
{
    if (raw_message == nullptr) {
        return;
    }

    unitree::common::REMOTE_DATA_RX remote{};
    remote.RF_RX.btn.value = static_cast<const Message*>(raw_message)->keys();
    const bool chord_pressed =
        remote.RF_RX.btn.components.Select && remote.RF_RX.btn.components.A;
    if (chord_pressed && !last_chord_pressed_) {
        pending_toggles_.fetch_add(1);
    }
    last_chord_pressed_ = chord_pressed;
}

}  // namespace camera_viewer
