#pragma once

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

class UdpSocket
{
public:
    UdpSocket(const std::string& host, uint16_t port)
    {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) {
            throw std::runtime_error("failed to create UDP socket");
        }

        const int broadcast = 1;
        if (::setsockopt(fd_, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to enable UDP broadcast");
        }

        std::memset(&addr_, 0, sizeof(addr_));
        addr_.sin_family = AF_INET;
        addr_.sin_port = htons(port);

        if (::inet_pton(AF_INET, host.c_str(), &addr_.sin_addr) != 1) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("invalid UDP destination address: " + host);
        }
    }

    ~UdpSocket()
    {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    UdpSocket(const UdpSocket&) = delete;
    UdpSocket& operator=(const UdpSocket&) = delete;

    UdpSocket(UdpSocket&& other) noexcept
    : fd_(other.fd_), addr_(other.addr_)
    {
        other.fd_ = -1;
    }

    UdpSocket& operator=(UdpSocket&& other) noexcept
    {
        if (this == &other) {
            return *this;
        }

        if (fd_ >= 0) {
            ::close(fd_);
        }

        fd_ = other.fd_;
        addr_ = other.addr_;
        other.fd_ = -1;
        return *this;
    }

    bool send(const void* data, size_t size) const
    {
        if (fd_ < 0) {
            return false;
        }

        const auto sent = ::sendto(
            fd_,
            data,
            size,
            0,
            reinterpret_cast<const sockaddr*>(&addr_),
            sizeof(addr_));
        return sent == static_cast<ssize_t>(size);
    }

private:
    int fd_ = -1;
    sockaddr_in addr_{};
};
