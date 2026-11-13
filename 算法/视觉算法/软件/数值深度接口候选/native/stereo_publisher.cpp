#include "stereo_publisher.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace agtc {
namespace {
constexpr std::int64_t max_skew_ns = 25'000'000;
constexpr std::int64_t max_initial_age_ns = 750'000'000;
constexpr std::int64_t max_processed_age_ns = 1'500'000'000;
bool valid_text(const std::string& s) {
    if (s.empty()) return false;
    for (unsigned char c : s) if (c < 0x20 || c > 0x7e) return false;
    return true;
}
bool fresh(std::int64_t now, std::int64_t arrival, std::int64_t max_age) {
    return now >= 0 && arrival >= 0 && arrival <= now && now - arrival <= max_age;
}
} // namespace

StereoPublisher::StereoPublisher(StereoSgbm& stereo, DepthStore& store, std::string session,
                                 std::string clock_id, Clock clock)
    : stereo_(stereo), store_(store), session_id_(std::move(session)),
      monotonic_clock_id_(std::move(clock_id)), clock_(std::move(clock)) {
    if (!valid_text(session_id_) || !valid_text(monotonic_clock_id_) || !clock_)
        throw std::invalid_argument("valid session, clock domain and clock required");
}

PublishStatus StereoPublisher::process(WorkToken token, const TimedStereoFrame& left,
                                       const TimedStereoFrame& right) {
    std::lock_guard<std::mutex> lock(publish_mutex_);
    auto reject = [&](PublishStatus reason) { store_.discard_if_current(token); return reason; };
    auto sample_clock = [&]() {
        try { return clock_(); }
        catch (...) { store_.discard_if_current(token); throw; }
    };
    if (left.pts_ns < 0 || right.pts_ns < 0 || left.gray.empty() || right.gray.empty() ||
        left.gray.size() != right.gray.size() || left.gray.type() != CV_8UC1 || right.gray.type() != CV_8UC1 ||
        (left.gray.size() != cv::Size(640, 360) && left.gray.size() != cv::Size(3840, 2160)))
        return reject(PublishStatus::invalid_input);
    const auto skew = left.pts_ns >= right.pts_ns ? left.pts_ns - right.pts_ns : right.pts_ns - left.pts_ns;
    if (skew > max_skew_ns) return reject(PublishStatus::skewed);
    const auto started = sample_clock();
    if (!fresh(started, left.arrival_ns, max_initial_age_ns) || !fresh(started, right.arrival_ns, max_initial_age_ns))
        return reject(PublishStatus::stale);
    if (last_token_ && last_token_->epoch == token.epoch &&
        last_token_->calibration_generation == token.calibration_generation) {
        if (left.pts_ns == last_left_pts_ns_) return PublishStatus::duplicate;
        if (left.pts_ns < last_left_pts_ns_) return reject(PublishStatus::regressed);
    }
    StereoOutput result;
    try { result = stereo_.compute(left.gray, right.gray); }
    catch (...) { store_.discard_if_current(token); throw; }
    const auto published = sample_clock();
    if (published < started || !fresh(published, left.arrival_ns, max_processed_age_ns) ||
        !fresh(published, right.arrival_ns, max_processed_age_ns)) return reject(PublishStatus::stale);
    if (last_sequence_ >= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        return reject(PublishStatus::sequence_exhausted);
    try {
        DepthContext context;
        context.session_id = session_id_;
        context.monotonic_clock_id = monotonic_clock_id_;
        context.calibration_id = stereo_.calibration_sha256().substr(0, 12);
        context.K = stereo_.wire_intrinsics();
        context.left_pts_ns = left.pts_ns; context.right_pts_ns = right.pts_ns;
        context.left_arrival_mono_ns = left.arrival_ns; context.right_arrival_mono_ns = right.arrival_ns;
        // Body extrinsics, source-clock mapping and preprocessing identity remain
        // null until independently bound to this actual deployment.
        if (!result.depth.isContinuous() || !publish_depth_f32(store_, token, last_sequence_ + 1,
            published, result.depth.ptr<float>(), result.depth.total(), context)) return reject(PublishStatus::changed);
    } catch (...) { store_.discard_if_current(token); throw; }
    ++last_sequence_;
    last_token_ = token;
    last_left_pts_ns_ = left.pts_ns;
    return PublishStatus::ok;
}
} // namespace agtc
