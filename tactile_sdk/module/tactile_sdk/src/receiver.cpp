
#include "receiver.h"

#include <fmt/format.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>

namespace sharpa {
namespace tactile {

Receiver::Receiver(const std::vector<int>& channels,
                   std::string host_ip,
                   int host_port,
                   int allowed_pack_loss,
                   std::shared_ptr<ThreadPool> pool,
                   std::shared_ptr<InferEngine> engine,
                   bool decode_jpeg)
    : host_ip_(host_ip),
      host_port_(host_port),
      allowed_pack_loss_(allowed_pack_loss),
      pool_(pool),
      engine_(engine) {
    stream_parser_ =
        std::make_shared<StreamParser>(channels, allowed_pack_loss, decode_jpeg);
    buffer_.resize(kMaxBytes_);
}

int Receiver::start() {
    if (is_running_) {
        return static_cast<int>(TACTILE_ERROR_CODE::TACCTILE_SDK_ALREADY_RUN);
    }

    struct sockaddr_in server_addr;
    if ((sock_ = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        return static_cast<int>(TACTILE_ERROR_CODE::HOST_LISTEN_FAULT);
    }

    struct timeval timeout;
    timeout.tv_sec = 0;
    timeout.tv_usec = 500'000; /* default timeout 500 ms */
    if (setsockopt(sock_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
        return static_cast<int>(TACTILE_ERROR_CODE::HOST_IP_INVALID);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(host_port_);
    // server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if (inet_pton(AF_INET, host_ip_.c_str(), &server_addr.sin_addr) <= 0) {
        LOG_ERROR("Invalid IP address: {}:{}", host_ip_, host_port_);
        close(sock_);
        return static_cast<int>(TACTILE_ERROR_CODE::HOST_IP_INVALID);
    }

    if (bind(sock_, (const struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        if (errno == EADDRINUSE) {
            LOG_ERROR(
                "Port {}:{} is already in use. Please choose another port or stop the "
                "conflicting process.",
                host_ip_, host_port_);
            close(sock_);
            return static_cast<int>(TACTILE_ERROR_CODE::HOST_PORT_OCCUPIED);
        } else {
            LOG_ERROR("Try to reach sharpa tactile {}:{}, but bind() failed: {}",
                      host_ip_, host_port_, strerror(errno));
            close(sock_);
            return static_cast<int>(TACTILE_ERROR_CODE::HOST_LISTEN_FAULT);
        }
    }

    th_ = std::thread([this] { start_loop_(); });
    is_running_ = true;
    return static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR);
}

void Receiver::start_loop_() {
    while(true) {
        {
            std::unique_lock<std::mutex> _(lock_);
            if(is_quit_) break;
        }
        socklen_t client_len = sizeof(client_addr_);
        int recv_len = recvfrom(sock_, buffer_.data(), kMaxBytes_, 0, (struct sockaddr *)&client_addr_, &client_len);
        if (recv_len < 0) continue;   /* tiomeout or other recv error */
        // auto ret = stream_parser_->receive(buffer_);
        auto ret = stream_parser_->receive_multi(buffer_);
        if(!ret.has_value()) continue;
        pool_->enqueue([this, ret]{
            auto raw_frame = *ret;
            auto content = raw_frame->parse();
            engine_->enqueue(content);
        });
    }
}

bool Receiver::stop() {
    if(!is_running_) return false;
    {
        std::unique_lock<std::mutex> _(lock_);
        is_quit_ = true;
    }
    th_.join();
    is_quit_ = false;
    close(sock_);
    is_running_ = false;
    return true;
}

bool Receiver::is_running() {
    return is_running_;
}

StreamParser::StreamStatistic Receiver::summary() const {
    return stream_parser_->summary();
}

void Receiver::set_thread_pool(std::shared_ptr<ThreadPool> pool) {
    pool_ = pool;
}
}
}
