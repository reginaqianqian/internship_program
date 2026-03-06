
#include "board_cfg.h"
#include "util.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <system_error>
#include <vector>

#include <iostream>

namespace sharpa {
namespace tactile {

std::string parse_msgpack(const msgpack11::MsgPack& obj, int indent) {
    std::stringstream ss;
    std::string space(indent, ' ');
    if (obj.is_object()) {
        ss << space << "{\n";
        for (const auto& kv : obj.object_items()) {
            ss << space << "  \"" << kv.first.string_value() << "\": ";
            ss << parse_msgpack(kv.second, indent + 4);
        }
        ss << space << "}\n";
    } else if (obj.is_array()) {
        ss << space << "[\n";
        for (const auto& el : obj.array_items()) {
            ss << parse_msgpack(el, indent + 4);
        }
        ss << space << "]\n";
    } else if (obj.is_string()) {
        ss << "\"" << obj.string_value() << "\"\n";
    } else if (obj.is_int()) {
        ss << obj.int_value() << "\n";
    } else if (obj.is_number()) {
        ss << obj.number_value() << "\n";
    } else if (obj.is_bool()) {
        ss << (obj.bool_value() ? "true" : "false") << "\n";
    } else if (obj.is_null()) {
        ss << "null\n";
    } else {
        ss << "(unknown)\n";
    }
    return ss.str();
}

std::tuple<int, std::vector<uint8_t>> tcp_request(const std::string& ip,
                                                  int port,
                                                  const std::vector<uint8_t>& msg,
                                                  double timeout,
                                                  const std::string& file) {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        LOG_WARN("failed to create socket socket(), error: {}", strerror(client_socket));
        return {static_cast<int>(TACTILE_ERROR_CODE::TCP_SYSTEM_ERROR), {}};
    }
    /* set timeout */
    struct timeval tv;
    tv.tv_sec = static_cast<long>(timeout);
    tv.tv_usec = static_cast<long>((timeout - tv.tv_sec) * 1000000);
    setsockopt(client_socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_socket, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    /* connect to server */
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr);

    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        LOG_WARN("failed to connect to sharpa device, device IP: {}", ip);
        close(client_socket);
        return {static_cast<int>(TACTILE_ERROR_CODE::TCP_CONNECT_FAILED), {}};
    }
    /* send message */
    if (send(client_socket, msg.data(), msg.size(), 0) < 0) {
        LOG_WARN("failed to send msg");
        close(client_socket);
        return {static_cast<int>(TACTILE_ERROR_CODE::TCP_SEND_ERROR), {}};
    }

    /* send file if specified */
    if (!file.empty()) {
        std::ifstream input_file(file, std::ios::binary);
        if (input_file) {
            input_file.seekg(0, std::ios::end);
            size_t file_size = input_file.tellg();
            input_file.seekg(0, std::ios::beg);

            char buffer[1024];
            size_t total_sent = 0;
            while (total_sent < file_size) {
                input_file.read(buffer, sizeof(buffer));
                ssize_t sent = send(client_socket, buffer, input_file.gcount(), 0);
                if (sent < 0) {
                    close(client_socket);
                    return {static_cast<int>(TACTILE_ERROR_CODE::TCP_SEND_ERROR), {}};
                }
                total_sent += sent;
            }
        }
    }

    /* receive data */
    std::vector<uint8_t> recv_data;
    char recv_buffer[1024];
    while (true) {
        ssize_t bytes_received = recv(client_socket, recv_buffer, sizeof(recv_buffer), 0);
        if (bytes_received <= 0)
            break;
        recv_data.insert(recv_data.end(), recv_buffer, recv_buffer + bytes_received);
    }
    close(client_socket);

    return {static_cast<int>(TACTILE_ERROR_CODE::NO_ERROR), recv_data};
}

std::tuple<int, std::optional<msgpack11::MsgPack>> msgpack_request(
    const std::string& ip,
    int port,
    const msgpack11::MsgPack& msg,
    double timeout) {
    std::vector<uint8_t> header{0xbb, 0xee, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::string body = msg.dump();
    *(uint32_t*)(header.data() + 8) = body.size();
    std::vector<uint8_t> tail{0xff, 0xff, 0xff, 0xff};

    std::vector<uint8_t> joined;
    joined.insert(joined.end(), header.begin(), header.end());
    joined.insert(joined.end(), body.begin(), body.end());
    joined.insert(joined.end(), tail.begin(), tail.end());

    auto tcp_ret = tcp_request(ip, port, joined, timeout, "");
    int err_no = std::get<0>(tcp_ret);
    std::vector<uint8_t> recv = std::get<1>(tcp_ret);
    if (err_no != 0) {
        return {err_no, std::nullopt};
    }
    // it is expected not to be empty
    if (recv.empty())
        return {static_cast<int>(TACTILE_ERROR_CODE::TCP_RECV_ERROR), std::nullopt};
    // if(recv.empty()) return std::nullopt;

    int32_t res_size = *(int32_t*)(recv.data() + 8);
    std::vector<char> bytes(recv.data() + 12, recv.data() + 12 + res_size);
    std::string err;

    auto ret = msgpack11::MsgPack::parse(bytes.data(), bytes.size(), err);
    // std::cout << "content: " << ret["content"][0]["channel"].int_value() << "\n";
    // if(!err.empty()) return std::nullopt;
    if (!err.empty()) {
        return {static_cast<int>(TACTILE_ERROR_CODE::TCP_MSG_PARSE_FALSE), std::nullopt};
    }
    return {0, ret};
    // return msgpack11::MsgPack::object{{"ret_code", 1314}};
}
}  // namespace tactile
}  // namespace sharpa
