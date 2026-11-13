#include "stereo_pair_queue.h"
#include <limits>

namespace agtc {
namespace {
bool same(WorkToken a, WorkToken b) {
    return a.epoch == b.epoch && a.calibration_generation == b.calibration_generation;
}
bool fresh(std::int64_t now, std::int64_t arrival) {
    return now >= 0 && arrival >= 0 && arrival <= now && now - arrival <= 750'000'000;
}
} // namespace

StereoPairQueue::StereoPairQueue(DepthStore& store) : store_(store) {}
std::optional<WorkToken> StereoPairQueue::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) return {};
    store_.set_capture_running(true);
    token_ = store_.token();
    frames_[0].clear(); frames_[1].clear();
    running_ = true;
    return token_;
}
void StereoPairQueue::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) return;
    running_ = false;
    store_.set_capture_running(false);
    frames_[0].clear(); frames_[1].clear();
}
bool StereoPairQueue::push(WorkToken token, int camera, const cv::Mat& gray,
                           std::int64_t pts_ns, std::int64_t arrival_ns) {
    if (camera < 0 || camera > 1 || pts_ns < 0 || arrival_ns < 0 || gray.empty() || gray.type() != CV_8UC1 ||
        (gray.size() != cv::Size(640, 360) && gray.size() != cv::Size(3840, 2160))) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || !same(token, token_) || !same(token, store_.token())) return false;
    }
    // GStreamer buffers may be recycled after callback return; snapshot before
    // taking the queue lock, then recheck generation before insertion.
    cv::Mat snapshot = gray.clone();
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ || !same(token, token_) || !same(token, store_.token())) return false;
    auto& frames = frames_[camera];
    if (!frames.empty() && (arrival_ns < frames.back().arrival_ns || pts_ns < frames.back().pts_ns)) return false;
    frames.push_back({std::move(snapshot), pts_ns, arrival_ns});
    if (frames.size() > 6) frames.pop_front();
    return true;
}
std::optional<StereoPair> StereoPairQueue::select(std::int64_t now, bool require_sync) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ || !same(token_, store_.token()) || frames_[0].empty() || frames_[1].empty()) return {};
    for (auto l = frames_[0].rbegin(); l != frames_[0].rend(); ++l) {
        if (!fresh(now, l->arrival_ns)) continue;
        const TimedStereoFrame* best = nullptr;
        std::int64_t best_skew = std::numeric_limits<std::int64_t>::max();
        for (const auto& r : frames_[1]) {
            if (!fresh(now, r.arrival_ns)) continue;
            const auto skew = l->pts_ns >= r.pts_ns ? l->pts_ns - r.pts_ns : r.pts_ns - l->pts_ns;
            if (!best || skew < best_skew) { best = &r; best_skew = skew; }
        }
        if (best && (!require_sync || best_skew <= 25'000'000)) return StereoPair{token_, *l, *best};
    }
    return {};
}
} // namespace agtc
