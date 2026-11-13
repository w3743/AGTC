#include "vision_runtime.h"
#include <limits>

namespace agtc {
namespace {
std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
VisionRuntime::VisionRuntime(const std::string& calibration, const std::string& session,
                             const std::string& clock_id)
    : stereo_(calibration), capture_(store_), publisher_(stereo_, store_, session, clock_id, [] {
          return std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch()).count();
      }) {
    store_.set_geometry_valid(true);
}
VisionRuntime::~VisionRuntime() { stop(); }
bool VisionRuntime::start(GstElement* pipeline) {
    std::lock_guard<std::mutex> lock(lifecycle_);
    if (started_) return false;
    started_ = true;
    if (!capture_.start(pipeline)) { failed_ = true; return false; }
    running_ = true;
    try {
        capture_threads_[0] = std::thread(&VisionRuntime::capture_loop, this, 0);
        capture_threads_[1] = std::thread(&VisionRuntime::capture_loop, this, 1);
        compute_thread_ = std::thread(&VisionRuntime::compute_loop, this);
    } catch (...) {
        failed_ = true; running_ = false; capture_.stop(); wake_.notify_all();
        for (auto& thread : capture_threads_) if (thread.joinable()) thread.join();
        if (compute_thread_.joinable()) compute_thread_.join();
        throw;
    }
    return true;
}
void VisionRuntime::stop() {
    std::lock_guard<std::mutex> lock(lifecycle_);
    running_ = false;
    capture_.stop();
    wake_.notify_all();
    for (auto& thread : capture_threads_) if (thread.joinable()) thread.join();
    if (compute_thread_.joinable()) compute_thread_.join();
}
void VisionRuntime::fail() {
    failed_ = true;
    running_ = false;
    capture_.stop();
    wake_.notify_all();
}
void VisionRuntime::request_depth() {
    const auto now = now_ns();
    if (now < 0) return;
    constexpr std::int64_t ttl = 5'000'000'000LL;
    const auto until = now > std::numeric_limits<std::int64_t>::max() - ttl ?
        std::numeric_limits<std::int64_t>::max() : now + ttl;
    auto previous = depth_until_ns_.load();
    while (previous < until && !depth_until_ns_.compare_exchange_weak(previous, until)) {}
    wake_.notify_one();
}
void VisionRuntime::capture_loop(int camera) {
    try {
        while (running_) {
            if (capture_.poll(camera, std::chrono::milliseconds(200))) {
                if (now_ns() <= depth_until_ns_.load()) wake_.notify_one();
            }
            else if (!capture_.is_running() && running_) { fail(); break; }
        }
    } catch (...) { fail(); }
}
void VisionRuntime::compute_loop() {
    try {
        while (running_) {
            const auto now = now_ns();
            const auto until = depth_until_ns_.load();
            if (until == 0 || now < 0 || now > until) {
                std::unique_lock<std::mutex> lock(wake_mutex_);
                wake_.wait_for(lock, std::chrono::milliseconds(200));
                continue;
            }
            auto pair = capture_.select();
            if (!capture_.is_running() && running_) { fail(); break; }
            if (pair) publisher_.process(pair->token, pair->left, pair->right);
            std::unique_lock<std::mutex> lock(wake_mutex_);
            wake_.wait_for(lock, std::chrono::milliseconds(20));
        }
    } catch (...) { fail(); }
}
} // namespace agtc
