#include "depth_http.h"
#include "depth_producer.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <limits>

using namespace std::chrono_literals;
int main() {
    agtc::DepthStore store(3);
    store.set_capture_running(true);
    store.set_geometry_valid(true);
    auto token = store.token();
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::vector<float> pixels(131 * 240, 1.0f);
    agtc::DepthContext context; context.session_id = "test";
    assert(agtc::publish_depth_f32(store, token, 1, now - 2'000'000'000LL, pixels.data(), pixels.size(), context));
    auto warming = agtc::route_depth_get(store, "/depth.bin", 0ns, 1s);
    assert(warming.status == 503 && warming.error_body.find("warming") != std::string::npos);
    auto historical = agtc::route_depth_get(store, "/depth.bin?sequence=%31", 0ns, 1s);
    assert(historical.status == 200 && historical.frame && historical.frame->sequence == 1);
    assert(historical.headers.find("Content-Length: 62880\r\n") != std::string::npos);
    assert(agtc::route_depth_get(store, "/depth.bin?sequence=1&sequence=1", 0ns, 1s).status == 400);
    int wakes = 0;
    auto wake = [&] { ++wakes; };
    assert(agtc::route_depth_get(store, "/depth.bin?sequence=bad", 0ns, 1s, wake).status == 400);
    assert(wakes == 0);
    assert(agtc::route_depth_get(store, "/depth.bin?sequence=1", 0ns, 1s, wake).status == 200);
    assert(wakes == 1);
    assert(agtc::route_depth_get(store, "/depth.bin?sequence=%0d", 0ns, 1s).status == 400);
    assert(agtc::route_depth_get(store, "/other", 0ns, 1s).status == 404);
    assert(agtc::route_depth_get(store, std::string(8193, 'x'), 0ns, 1s).status == 400);
    assert(store.publish(token, agtc::DepthFrame{2, token.epoch, now, {0, 1}, "{}"}));
    assert(agtc::route_depth_get(store, "/depth.bin", 0ns, 1s).status == 503);
    assert(store.publish(token, agtc::DepthFrame{3, token.epoch, now,
        std::vector<std::uint8_t>(131 * 240 * 2), "{}\r\nInjected: yes"}));
    auto bad = agtc::route_depth_get(store, "/depth.bin", 0ns, 1s);
    assert(bad.status == 503 && bad.headers.find("Injected:") == std::string::npos);
}
