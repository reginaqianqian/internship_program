
#include <catch2/catch_all.hpp>

#include <touch.h>
#include <unistd.h>
#include <iostream>

using namespace sharpa::tactile;

TEST_CASE("Frame", "[tactile_sdk]") {
    Frame f;
    auto db = std::make_shared<DataBlock>(Shape{{2, 3}}, sizeof(int));
    db->as<int>().at({0, 0}) = 23;
    f.content.emplace("foo", db);

    auto f_copy = f.copy();
    CHECK(f_copy->frame_id == f.frame_id);
    CHECK(f_copy->content["foo"]->as<int>().at({0, 0}) == 23);

    /* check DataBlock is indeed copied */
    f_copy->content["foo"]->as<int>().at({0, 0}) = 29;
    CHECK(db->as<int>().at({0, 0}) == 23);
}

TEST_CASE("touch", "[tactile_sdk]") {
    auto touch = std::make_shared<Touch>(
        "0.0.0.0"
        , 50001
        , std::vector<int>{0, 1, 2, 3, 4}
        , std::vector<std::string>{"192.168.10.20"}
        , std::nullopt
        , nullptr
    );

    /* test load mapping */
    auto uv_zero = touch->deform_map_uv(0, 0, 0);
    CHECK(!uv_zero.has_value());
    auto uv_nonzero = touch->deform_map_uv(0, 120, 120);  /* center point, must have value */
    CHECK(uv_nonzero.has_value());

    /* test deform value mapping */
    CHECK(touch->deform_map_value(0) == 0.f);
    CHECK(touch->deform_map_value(100) == 0.5f);
    CHECK(touch->deform_map_value(101) == 0.5f + 3e-2f);

    float nx = uv_nonzero->at(3);
    float ny = uv_nonzero->at(4);
    float nz = uv_nonzero->at(5);
    CHECK(1 == Catch::Approx(nx * nx + ny * ny + nz * nz).epsilon(1e-3f));  /* normal is normalized */

    for (int i = 0; i < 30; ++i) {
        touch->set_callback([](Frame::Ptr f) {});
        if (i % 10 == 0)
            std::cerr << "restart " << i << " times\n";
        while (touch->start() != 0) {
            usleep(500'000);
        };
        usleep(500'000);
        touch->stop();
    }
}
