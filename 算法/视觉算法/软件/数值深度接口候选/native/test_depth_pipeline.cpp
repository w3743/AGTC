#include "depth_producer.h"
#include "depth_request.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

using namespace std::chrono_literals;
int main(int argc, char** argv) {
    agtc::DepthStore store(2);
    store.set_capture_running(true);
    store.set_geometry_valid(true);
    const auto token = store.token();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::vector<float> pixels(131 * 240, 1.25f);
    pixels[0] = 0.0f; pixels[1] = 0.3f; pixels[2] = 5.0f;
    pixels[3] = std::numeric_limits<float>::quiet_NaN();
    agtc::DepthContext context;
    context.session_id = "unit-test\"session";
    context.monotonic_clock_id = "test-clock";
    context.left_pts_ns = 9007199254740993LL;
    context.K = std::array<double, 9>{100, 0, 60, 0, 101, 120, 0, 0, 1};
    assert(!agtc::publish_depth_f32(store, token, 0, now, pixels.data(), pixels.size(), context));
    assert(!agtc::publish_depth_f32(store, token, 1, now, pixels.data(), pixels.size() - 1, context));
    auto bad = context; bad.session_id = "bad\r\nHeader: value";
    assert(!agtc::publish_depth_f32(store, token, 1, now, pixels.data(), pixels.size(), bad));
    bad = context; bad.K = std::array<double, 9>{-1, 0, 0, 0, 1, 0, 0, 0, 1};
    assert(!agtc::publish_depth_f32(store, token, 1, now, pixels.data(), pixels.size(), bad));
    assert(agtc::publish_depth_f32(store, token, 1, now, pixels.data(), pixels.size(), context));
    for (const auto& s : {"", "0", "-1", "+1", "1x", "9223372036854775808", "00000000000000000001"})
        assert(agtc::handle_depth_request(store, std::vector<std::string>{s}, 0ns, 1s).status == 400);
    assert(agtc::handle_depth_request(store, std::vector<std::string>{"1", "2"}, 0ns, 1s).status == 400);
    assert(agtc::handle_depth_request(store, std::vector<std::string>{"2"}, 0ns, 1s).status == 409);
    auto response = agtc::handle_depth_request(store, std::nullopt, 0ns, 1s);
    assert(response.status == 200 && response.frame && response.frame->sequence == 1);
    assert(agtc::handle_depth_request(store, std::vector<std::string>{"001"}, 0ns, 1s).status == 200);
    if (argc == 2) {
        std::ofstream file(argv[1], std::ios::binary);
        file.write(reinterpret_cast<const char*>(response.frame->payload.data()),
                   static_cast<std::streamsize>(response.frame->payload.size()));
        if (!file) return 2;
        std::cout << response.frame->metadata_ascii << '\n';
    }
    std::vector<float> disparity(131 * 240, 10.0f);
    std::vector<std::uint8_t> support(disparity.size(), 1);
    assert(agtc::publish_disparity_f32(store, token, 2, now, disparity.data(),
        support.data(), support.data(), disparity.size(), 0.0, 10.0, context));
    auto filtered = agtc::handle_depth_request(store, std::vector<std::string>{"2"}, 0ns, 1s);
    assert(filtered.status == 200 && filtered.frame->payload[0] == 0 && filtered.frame->payload[1] == 0);
    auto at = (131 + 12) * 2;
    assert(filtered.frame->payload[at] == 0xe8 && filtered.frame->payload[at + 1] == 0x03);
    store.set_capture_running(false);
    assert(!agtc::publish_depth_f32(store, token, 3, now, pixels.data(), pixels.size(), context));
    assert(agtc::handle_depth_request(store, std::nullopt, 0ns, 1s).status == 503);
}
