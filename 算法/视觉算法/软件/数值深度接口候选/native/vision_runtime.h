#pragma once
#include "gst_stereo_capture.h"
#include <atomic>
#include <condition_variable>
#include <thread>

namespace agtc {
// A single-use capture/compute runtime. The caller owns the pipeline reference,
// starts gst_init first, and keeps this object alive until HTTP workers stop.
// No automatic restart, real-time or vehicle-safety guarantee is implied.
class VisionRuntime {
public:
    VisionRuntime(const std::string& calibration_yaml, const std::string& session_id,
                  const std::string& monotonic_clock_id);
    ~VisionRuntime();
    VisionRuntime(const VisionRuntime&) = delete;
    VisionRuntime& operator=(const VisionRuntime&) = delete;
    bool start(GstElement* pipeline);
    void stop();
    bool running() const { return running_.load(); }
    bool failed() const { return failed_.load(); }
    // Request-driven depth compute is active for five seconds since last request.
    void request_depth();
    // For display only; the pair may exceed the depth pipeline skew bound.
    std::optional<StereoPair> preview_pair() { return capture_.select(false); }
    DepthStore& store() { return store_; }
private:
    void fail();
    void capture_loop(int camera);
    void compute_loop();
    DepthStore store_{6};
    StereoSgbm stereo_;
    GstStereoCapture capture_;
    StereoPublisher publisher_;
    std::mutex lifecycle_, wake_mutex_;
    std::condition_variable wake_;
    std::thread capture_threads_[2], compute_thread_;
    std::atomic<bool> running_{false}, failed_{false};
    std::atomic<std::int64_t> depth_until_ns_{0};
    bool started_ = false;
};
} // namespace agtc
