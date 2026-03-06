#pragma once

#include <msgpack11.hpp>
#include <variant>

#include <def.h>

namespace sharpa {
namespace tactile {

std::tuple<int, std::vector<uint8_t>> tcp_request(const std::string& ip,
                                                  int port,
                                                  const std::vector<uint8_t>& msg,
                                                  double timeout,
                                                  const std::string& file);

std::tuple<int, std::optional<msgpack11::MsgPack>> msgpack_request(
    const std::string& ip,
    int port,
    const msgpack11::MsgPack& msg,
    double timeout);

std::string parse_msgpack(const msgpack11::MsgPack& obj, int indent = 0);

}
}
