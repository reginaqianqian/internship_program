

#pragma once

#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>

#include "infer.h"
#include "stream_protocol.h"
#include "util.h"

namespace sharpa {
namespace tactile {

class Receiver {
public:
    Receiver(const std::vector<int>& channels,
             std::string host_ip,
             int host_port,
             int allowed_pack_loss = 4,
             std::shared_ptr<ThreadPool> pool = nullptr,
             std::shared_ptr<InferEngine> engine = nullptr,
             bool decode_jpeg = false);

    int start();

    void start_loop_();

    bool stop();

    bool is_running();

    void set_thread_pool(std::shared_ptr<ThreadPool> pool);

    StreamParser::StreamStatistic summary() const;

private:
    std::string host_ip_;
    int host_port_;
    int allowed_pack_loss_;
    std::shared_ptr<ThreadPool> pool_;
    std::shared_ptr<InferEngine> engine_;
    int sock_{-1};
    std::shared_ptr<StreamParser> stream_parser_;
    bool is_running_{false};
    std::thread th_;
    std::mutex lock_;
    bool is_quit_{false};
    const int kMaxBytes_{1472};
    std::vector<uint8_t> buffer_;

    struct sockaddr_in client_addr_;
};

}  // namespace tactile
}  // namespace sharpa
