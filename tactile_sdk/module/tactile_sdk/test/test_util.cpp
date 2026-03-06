#include <catch2/catch_all.hpp>

#include <src/util.h> 

using namespace sharpa::tactile;

TEST_CASE("SafeQueue", "[tactile_sdk]") {
    SafeQueue<int> queue(2);
    queue.enqueue(1);
    queue.enqueue(2);
    queue.enqueue(3);
    CHECK(queue.dequeue().value() == 2);
    CHECK(queue.size() == 1);
    CHECK(queue.dequeue().value() == 3);
    CHECK(!queue.dequeue(0.01).has_value());
    /* TO_DO test multi-thread safety */
}