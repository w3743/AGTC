// Host microbenchmark fixture; no camera or network involved.
#include "depth_preview.h"
#include <opencv2/core.hpp>
#include <chrono>
#include <iostream>

int main() {
    cv::setNumThreads(2);
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    store.set_capture_running(true);
    auto token = store.token();
    agtc::DepthPreview preview(store);
    std::vector<std::uint8_t> payload(131 * 240 * 2);
    for (std::size_t i = 0; i < payload.size() / 2; ++i) {
        unsigned mm = i % 7 == 0 ? 0 : 300 + static_cast<unsigned>(i % 4701);
        payload[2 * i] = static_cast<std::uint8_t>(mm & 255);
        payload[2 * i + 1] = static_cast<std::uint8_t>(mm >> 8);
    }
    for (int i = 0; i < 550; ++i) {
        auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (!store.publish(token, {static_cast<std::uint64_t>(i + 1), token.epoch, now, payload, "{}"})) return 1;
        auto start = std::chrono::steady_clock::now();
        auto frame = preview.latest(std::chrono::nanoseconds(0), std::chrono::seconds(60));
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        if (frame.status != agtc::DepthStatus::ok || !frame.frame) return 2;
        if (i >= 50) std::cout << "fresh " << elapsed << '\n';
    }
    for (int i = 0; i < 550; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto frame = preview.latest(std::chrono::nanoseconds(0), std::chrono::seconds(60));
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        if (frame.status != agtc::DepthStatus::ok || !frame.frame) return 3;
        if (i >= 50) std::cout << "cached " << elapsed << '\n';
    }
}
