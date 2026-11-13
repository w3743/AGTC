#pragma once
#include "depth_producer.h"
#include "stereo_sgbm.h"
#include <functional>

namespace agtc {
struct TimedStereoFrame {
    cv::Mat gray;
    std::int64_t pts_ns;
    std::int64_t arrival_ns;
};
enum class PublishStatus { ok, duplicate, regressed, invalid_input, skewed, stale, changed, sequence_exhausted };

// PTS values must be in a common capture-pipeline time domain. Arrival and
// clock values must be in DepthStore's steady_clock domain; neither is an
// exposure timestamp. Caller binds both images to the supplied work token.
// This object serializes matcher use and sequence assignment. It must be the
// sole publisher for its store, and both must share one process lifetime.
class StereoPublisher {
public:
    using Clock = std::function<std::int64_t()>;
    StereoPublisher(StereoSgbm& stereo, DepthStore& store, std::string session_id,
                    std::string monotonic_clock_id, Clock clock);
    PublishStatus process(WorkToken token, const TimedStereoFrame& left,
                          const TimedStereoFrame& right);
private:
    StereoSgbm& stereo_;
    DepthStore& store_;
    std::string session_id_, monotonic_clock_id_;
    Clock clock_;
    std::mutex publish_mutex_;
    std::uint64_t last_sequence_ = 0;
    std::optional<WorkToken> last_token_;
    std::int64_t last_left_pts_ns_ = 0;
};
} // namespace agtc
