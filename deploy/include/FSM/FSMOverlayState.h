#pragma once

#include <string>

class FSMOverlayState
{
public:
    virtual ~FSMOverlayState() = default;

    virtual std::string name() const = 0;
    virtual bool should_activate() const = 0;
    virtual void on_activate(const std::string& host_state_name) = 0;
    virtual void on_update() = 0;
    virtual void on_deactivate() = 0;

    void activate(const std::string& host_state_name)
    {
        active_ = true;
        finished_ = false;
        requested_state_id_ = 0;
        on_activate(host_state_name);
    }

    void update()
    {
        if (!active_)
        {
            return;
        }
        on_update();
    }

    void deactivate()
    {
        if (!active_)
        {
            return;
        }
        on_deactivate();
        active_ = false;
    }

    bool active() const
    {
        return active_;
    }

    bool finished() const
    {
        return finished_;
    }

    int requested_state_id() const
    {
        return requested_state_id_;
    }

protected:
    void finish(int requested_state_id = 0)
    {
        finished_ = true;
        requested_state_id_ = requested_state_id;
    }

private:
    bool active_{false};
    bool finished_{false};
    int requested_state_id_{0};
};
