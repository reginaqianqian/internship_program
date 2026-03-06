
#include <catch2/catch_all.hpp>
#include <iostream>

#include <src/board_cfg.h>

using namespace sharpa::tactile;

TEST_CASE("msgpack", "[tactile_sdk]") {
    msgpack11::MsgPack my_msgpack = msgpack11::MsgPack::object {
        { "key1", "value1" },
        { "key2", false },
        { "key3", msgpack11::MsgPack::array { 1, 2, 3 } },
    };

    CHECK(my_msgpack["key1"].string_value() == "value1");

    std::string msgpack_bytes = my_msgpack.dump();
    std::string err;
    msgpack11::MsgPack des_msgpack = msgpack11::MsgPack::parse(msgpack_bytes, err);

    CHECK(des_msgpack["key3"][0].int_value() == 1);
    CHECK(des_msgpack["key_3.14"].is_null());

    int int_value{23};
    msgpack11::MsgPack msgpack_int{int_value};
    CHECK(!msgpack_int.is_float64());
    CHECK(!msgpack_int.is_int16());

    msgpack11::MsgPack obj = msgpack11::MsgPack::object{
        {"value", 0.f},
        {"int_value", 0},
    };
    std::string str = obj.dump();
    CHECK(str.size() == 23);
}
