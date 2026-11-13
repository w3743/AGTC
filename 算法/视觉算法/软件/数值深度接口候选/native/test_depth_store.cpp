#include "depth_store.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace std::chrono_literals;
using agtc::DepthStatus;

static std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
static agtc::DepthFrame frame(agtc::WorkToken token, std::uint64_t sequence, std::int64_t time) {
    return {sequence, token.epoch, time, {1, 2, 3}, "{}"};
}

int main() {
    bool threw = false;
    try { agtc::DepthStore bad(0); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    agtc::DepthStore store(2);
    assert(store.read({}, 0ns, 1s).status == DepthStatus::unavailable);
    store.set_capture_running(true);
    store.set_geometry_valid(true);
    auto first = store.token();
    assert(store.publish(first, frame(first, 1, now_ns())));
    auto snapshot = store.read({}, 0ns, 1s);
    assert(snapshot.status == DepthStatus::ok && snapshot.frame->sequence == 1);
    assert(!store.publish(first, frame(first, 1, now_ns())));
    assert(store.publish(first, frame(first, 2, now_ns())));
    assert(store.publish(first, frame(first, 3, now_ns())));
    assert(store.read(1, 0ns, 1s).status == DepthStatus::expired);
    assert(snapshot.frame->sequence == 1 && snapshot.frame->payload[2] == 3);
    assert(store.publish(first, frame(first, 4, now_ns() - 2'000'000'000LL)));
    assert(store.read({}, 0ns, 1s).status == DepthStatus::warming);
    assert(store.read(4, 0ns, 1s).status == DepthStatus::ok);
    assert(store.publish(first, frame(first, 5, now_ns() + 10'000'000'000LL)));
    assert(store.read({}, 0ns, 1s).status == DepthStatus::warming);

    auto pending = std::async(std::launch::async, [&] { return store.read(6, 2s, 1s); });
    assert(pending.wait_for(50ms) == std::future_status::timeout);
    assert(store.publish(first, frame(first, 6, now_ns())));
    assert(pending.get().status == DepthStatus::ok);

    pending = std::async(std::launch::async, [&] { return store.read(7, 2s, 1s); });
    assert(pending.wait_for(50ms) == std::future_status::timeout);
    store.set_capture_running(false);
    assert(pending.get().status == DepthStatus::changed);
    assert(!store.publish(first, frame(first, 7, now_ns())));
    store.set_capture_running(true);
    auto second = store.token();
    pending = std::async(std::launch::async, [&] { return store.read(8, 2s, 1s); });
    assert(pending.wait_for(50ms) == std::future_status::timeout);
    store.set_calibrating(true);
    assert(pending.get().status == DepthStatus::calibrating);
    assert(store.read({}, 0ns, 1s).status == DepthStatus::calibrating);
    store.set_calibrating(false);
    assert(!store.publish(second, frame(second, 8, now_ns())));
    auto third = store.token();
    pending = std::async(std::launch::async, [&] { return store.read(9, 2s, 1s); });
    assert(pending.wait_for(50ms) == std::future_status::timeout);
    store.set_geometry_valid(false);
    assert(pending.get().status == DepthStatus::changed);
    store.set_geometry_valid(true);
    assert(!store.publish(third, frame(third, 9, now_ns())));
    const auto fourth = store.token();
    assert(store.publish(fourth, frame(fourth, 10, now_ns())));
    assert(!store.discard_if_current(third));
    assert(store.read({}, 0ns, 1s).status == DepthStatus::ok);
    assert(store.discard_if_current(fourth));
    assert(store.read({}, 0ns, 1s).status == DepthStatus::unavailable);
    std::cout << "depth store checks passed\n";
}
