
#include <catch2/catch_all.hpp>

#include <tensor.h>
#include <memory>

using namespace sharpa::tactile;

TEST_CASE("range", "[tactile_sdk]") {
    Range range(2, 12);
    CHECK(range.l() == 2);
    CHECK(range.r() == 12);
    CHECK(range.inside(2));
    CHECK(range.inside(10));
    CHECK(!range.inside(12));

    REQUIRE_THROWS_MATCHES(
        Range(4, 4), 
        std::runtime_error, 
        Catch::Matchers::Message("bad range")
    );

    Range range_cp(range);
    CHECK(range_cp.l() == 2);
    CHECK(range_cp.r() == 12);
    
    Range range_mv(std::move(range));
    CHECK(range_mv.l() == 2);
    CHECK(range_mv.r() == 12);
    CHECK(range.l() == 2);
    CHECK(range.r() == 12);
}

TEST_CASE("slice", "[tactile_sdk]") {
    Slice slice({
        {0, 1},
        {10, 90},
        {4, 8},
    });
    REQUIRE_THROWS_MATCHES(
        Slice({{0, 1}, {9, 3}}), 
        std::runtime_error, 
        Catch::Matchers::Message("bad range")
    );

    CHECK(slice.dim() == 3);
    CHECK(slice.inside({0, 32, 5}));
    CHECK(!slice.inside({1, 32, 5}));
    CHECK(!slice.inside({0, 9, 5}));
    CHECK(!slice.inside({0, 32, 8}));
    REQUIRE_THROWS_MATCHES(
        slice.inside({0, 32}), 
        std::runtime_error, 
        Catch::Matchers::Message("dim not equal")
    );

    CHECK(slice.idx_abs({0, 12, 2}) == Index{0, 22, 6});
    REQUIRE_THROWS_MATCHES(
        slice.idx_abs({0, 80, 2}), 
        std::runtime_error, 
        Catch::Matchers::Message("idx out of range")
    );

    Slice slice_cp(slice);
    CHECK(slice.shape() == slice_cp.shape());

    Slice slice_mv(std::move(slice));
    CHECK(slice_mv.shape() == slice_cp.shape());
    CHECK(slice.dim() == 0);
}

TEST_CASE("shape", "[tactile_sdk]") {
    Shape s({2, 320, 240});
    
    CHECK(s.size() == 2 * 320 * 240);
    CHECK(s.dim() == 3);
    CHECK(s.flat_idx({1, 1, 1}) == 1 * 320 * 240 + 1 * 240 + 1);
    CHECK(s.flat_idx({1, 20, 17}) == 1 * 320 * 240 + 20 * 240 + 17);
    REQUIRE_THROWS_MATCHES(
        s.flat_idx({0, 0}), 
        std::runtime_error, 
        Catch::Matchers::Message("dim not equal")
    );
    REQUIRE_THROWS_MATCHES(
        s.flat_idx({2, 0, 0}), 
        std::runtime_error, 
        Catch::Matchers::Message("idx out of range")
    );
    CHECK(s.flat_idx({1, 20, 17}) == 1 * 320 * 240 + 20 * 240 + 17);
    CHECK(s.inside({1, 319, 239}));
    CHECK(!s.inside({1, 320, 239}));

    Shape s_cp(s);
    CHECK(s_cp == s);

    Shape s_mv(std::move(s));
    CHECK(s_mv == s_cp);
    CHECK(s.dim() == 0);
    CHECK(s.size() == 0);

    /* test all_indices() */
    Shape s2({1, 2, 3});
    auto ids = s2.all_indices();
    CHECK(ids.size() == 1 * 2 * 3);
    CHECK(std::equal(ids[0].begin(), ids[0].end(), Index{0, 0, 0}.begin()));
    CHECK(std::equal(ids[1].begin(), ids[1].end(), Index{0, 0, 1}.begin()));
    CHECK(std::equal(ids[2].begin(), ids[2].end(), Index{0, 0, 2}.begin()));
    CHECK(std::equal(ids[3].begin(), ids[3].end(), Index{0, 1, 0}.begin()));
    CHECK(std::equal(ids[4].begin(), ids[4].end(), Index{0, 1, 1}.begin()));
    CHECK(std::equal(ids[5].begin(), ids[5].end(), Index{0, 1, 2}.begin()));
}

TEST_CASE("DataBlock;Tensor", "[tactile_sdk]") {
    /* test empty tensor */
    DataBlock db_empty({}, sizeof(double));
    CHECK(db_empty.size() == 0);
    CHECK(db_empty.nbytes() == 0);
    CHECK(db_empty.shape() == Shape{});
    CHECK(db_empty.dim() == 0);

    /* test tensor */
    DataBlock db({{2, 240, 320}}, sizeof(float));
    CHECK(db.size() == 2 * 240 * 320);
    CHECK(db.nbytes() == 2 * 240 * 320 * sizeof(float));
    CHECK(db.shape() == Shape{{2, 240, 320}});
    CHECK(db.dim() == 3);
    
    REQUIRE_THROWS_MATCHES(
        db.as<double>(),
        std::runtime_error, 
        Catch::Matchers::Message("unit size dismatch")
    );

    DataBlock db_cp(db);
    CHECK(db_cp.shape() == db.shape());
    CHECK(db_cp.size() == db.size());
    CHECK(db_cp.nbytes() == db.nbytes());
    CHECK(((int *)db_cp.data())[317] == ((int *)db.data())[317]);

    DataBlock db_mv(std::move(db_cp));
    CHECK(db_mv.shape() == db.shape());
    CHECK(db_mv.size() == db.size());
    CHECK(db_mv.nbytes() == db.nbytes());
    CHECK(((int *)db_mv.data())[317] == ((int *)db.data())[317]);

    CHECK(db_cp.shape() == Shape{{}});

    auto tensor = db.as<float>();
    CHECK(tensor.shape() == Shape{{2, 240, 320}});
    CHECK(tensor.size() == 2 * 240 * 320);
    CHECK(tensor.dim() == 3);

    memset(db.data(), 0, db.nbytes());    

    for(size_t i = 0; i < tensor.shape()[0]; ++i)
    for(size_t j = 0; j < tensor.shape()[1]; ++j)
    for(size_t k = 0; k < tensor.shape()[2]; ++k)
        CHECK(tensor.at({i, j, k}) == 0.f);

    REQUIRE_THROWS_MATCHES(
        tensor.at({2, 0, 0}),
        std::runtime_error, 
        Catch::Matchers::Message("idx out of range")
    );

    /* test slice */
    Slice s({{0, 2}, {10, 30}, {100, 200}});
    auto slice = tensor.slice(s);
    CHECK(slice.size() == 2 * 20 * 100);
    CHECK(slice.dim() == 3);

    for(const auto &idx : slice.shape().all_indices())
        slice.at(idx) = 1.f;

    for(const auto &idx : tensor.shape().all_indices()) {
        if(s.inside(idx)) CHECK(tensor.at(idx) == 1.f);
        else CHECK(tensor.at(idx) == 0.f);
    }

    /* test nested slice */
    Slice s2({{0, 1}, {0, 5}, {0, 50}});
    auto slice2 = slice.slice(s2);

    for(const auto &idx : slice2.shape().all_indices())
        slice2.at(idx) = 2.f;
    
    for(const auto &idx : slice.shape().all_indices()) {
        if(s2.inside(idx)) CHECK(slice.at(idx) == 2.f);
        else CHECK(slice.at(idx) == 1.f);
    }

    /* test assign */
    slice2.assign(slice.slice({{{1, 2}, {0, 5}, {0, 50}}}));

    for(const auto &idx : slice2.shape().all_indices())
        CHECK(slice2.at(idx) == 1.f);
    
    /* test math operations */
    auto x = std::make_shared<DataBlock>(Shape{{1, 2}}, sizeof(float));
    auto y = std::make_shared<DataBlock>(Shape{{1, 2}}, sizeof(float));
    auto tx = x->as<float>();
    auto ty = y->as<float>();
    tx.at({0, 0}) = 1.f;
    tx.at({0, 1}) = -1.f;
    ty.at({0, 0}) = 2.f;
    ty.at({0, 1}) = 3.f;

    auto sum = add_db_float(x, y);
    for(int _ = 0; _ < 20; ++_) {
        sum = add_db_float(sum, y);
    }
    auto t_sum = sum->as<float>();
    CHECK(t_sum.at({0, 0}) == 43.f);
    CHECK(t_sum.at({0, 1}) == 62.f);

    auto diff = sub_db_float(x, y);
    auto t_diff = diff->as<float>();
    CHECK(t_diff.at({0, 0}) == -1.f);
    CHECK(t_diff.at({0, 1}) == -4.f);

    auto mul = mul_db_float(x, 0.5f);
    auto t_mul = mul->as<float>();
    CHECK(t_mul.at({0, 0}) == 0.5f);
    CHECK(t_mul.at({0, 1}) == -0.5f);

    mul->set_zero();
    CHECK(t_mul.at({0, 0}) == 0.f);
    CHECK(t_mul.at({0, 1}) == 0.f);

    auto f32 = std::make_shared<DataBlock>(Shape{{2, 2}}, sizeof(float));
    auto tf32 = f32->as<float>();
    tf32.at({0, 0}) = -0.2f;
    tf32.at({0, 1}) = 1.5f;
    tf32.at({1, 0}) = 34.4f;
    tf32.at({1, 1}) = 256.9f;

    auto i8 = db_f32_to_ui8(f32);
    CHECK(i8->nbytes() * 4 == f32->nbytes());
    auto ti8 = i8->as<uint8_t>();
    CHECK(ti8.at({0, 0}) == 0);
    CHECK(ti8.at({0, 1}) == 2);
    CHECK(ti8.at({1, 0}) == 34);
    CHECK(ti8.at({1, 1}) == 255);

    f32 = db_ui8_to_f32(i8);
    tf32 = f32->as<float>();
    CHECK(tf32.at({0, 0}) == 0.f);
    CHECK(tf32.at({0, 1}) == 2.f);
    CHECK(tf32.at({1, 0}) == 34.f);
    CHECK(tf32.at({1, 1}) == 255.f);

    /* test reshape */
    DataBlock db_reshape({{2, 3}}, sizeof(int));
    auto t_to_reshape = db_reshape.as<int>();
    t_to_reshape.at({0, 0}) = 0;
    t_to_reshape.at({0, 1}) = 1;
    t_to_reshape.at({0, 2}) = 2;
    t_to_reshape.at({1, 0}) = 3;
    t_to_reshape.at({1, 1}) = 4;
    t_to_reshape.at({1, 2}) = 5;

    db_reshape.reshape({{3, 2}});
    auto t_reshaped = db_reshape.as<int>();

    CHECK(t_reshaped.at({0, 0}) == 0);
    CHECK(t_reshaped.at({0, 1}) == 1);
    CHECK(t_reshaped.at({1, 0}) == 2);
    CHECK(t_reshaped.at({1, 1}) == 3);
    CHECK(t_reshaped.at({2, 0}) == 4);
    CHECK(t_reshaped.at({2, 1}) == 5);

    /* test load from txt */
    auto db_from_txt = f32_from_txt("../../../config/static/general_ha4_map_normal.txt", 240*240*3);
    CHECK(db_from_txt->shape() == Shape{{240 * 240 * 3}});
}
