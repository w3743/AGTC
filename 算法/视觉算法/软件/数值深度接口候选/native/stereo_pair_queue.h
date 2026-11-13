#pragma once
#include "stereo_publisher.h"
#include <deque>
#include <optional>

namespace agtc {
struct StereoPair {
    WorkToken token;
    TimedStereoFrame left;
    TimedStereoFrame right;
};

// One capture session owns this queue. start()/stop() must be used for all
// pipeline transitions; calibration changes require stopping and starting a
// new session. push() copies pixels while the caller still owns the buffer.
// Returned images are shared immutable snapshots: callers must not modify them.
class StereoPairQueue {
public:
    explicit StereoPairQueue(DepthStore& store);
    std::optional<WorkToken> start();
    void stop();
    bool push(WorkToken token, int camera, const cv::Mat& gray,
              std::int64_t pts_ns, std::int64_t arrival_ns);
    // Preview may opt out of the skew bound; freshness is always enforced.
    std::optional<StereoPair> select(std::int64_t now_ns, bool require_sync = true) const;
private:
    DepthStore& store_;
    mutable std::mutex mutex_;
    bool running_ = false;
    WorkToken token_{};
    std::deque<TimedStereoFrame> frames_[2];
};
} // namespace agtc
