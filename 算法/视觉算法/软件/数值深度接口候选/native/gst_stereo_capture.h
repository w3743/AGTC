#pragma once
#include "stereo_pair_queue.h"
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

namespace agtc {
// Returns a new pipeline reference or nullptr. Caller owns the returned reference.
// v4l2src must be available at runtime; device paths are passed as properties,
// not interpolated into a pipeline description.
GstElement* make_v4l2_stereo_pipeline(const std::string& left_device,
                                      const std::string& right_device,
                                      int width = 640, int height = 360);
// The supplied pipeline must contain appsinks named sink0 and sink1, delivering
// NV12 with a common GstSegment running-time clock. gst_init() is the caller's
// responsibility. poll() may run on two capture threads; stop() invalidates the
// generation before setting the pipeline NULL and releases owned references.
// Stop and join all callers of poll()/select() before destroying this object.
class GstStereoCapture {
public:
    explicit GstStereoCapture(DepthStore& store);
    ~GstStereoCapture();
    GstStereoCapture(const GstStereoCapture&) = delete;
    GstStereoCapture& operator=(const GstStereoCapture&) = delete;
    bool start(GstElement* pipeline);
    void stop();
    bool poll(int camera, std::chrono::milliseconds timeout);
    std::optional<StereoPair> select(bool require_sync = true);
    bool is_running() const;
private:
    void stop_if_current(WorkToken token);
    bool check_bus(WorkToken token, GstElement* pipeline);
    mutable std::mutex control_;
    StereoPairQueue queue_;
    GstElement* pipeline_ = nullptr;
    GstAppSink* sinks_[2] = {nullptr, nullptr};
    std::optional<WorkToken> token_;
};
} // namespace agtc
