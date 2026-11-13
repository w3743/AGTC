#include "depth_store.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace agtc {

DepthStore::DepthStore(std::size_t capacity) : capacity_(capacity) {
    if (capacity == 0) throw std::invalid_argument("depth history capacity must be positive");
}

void DepthStore::set_capture_running(bool running) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Each call denotes a pipeline transition, including a restart.
    running_ = running;
    ++epoch_;
    history_.clear();
    changed_.notify_all();
}

void DepthStore::set_calibrating(bool active) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (calibrating_ != active) {
        calibrating_ = active;
        ++calibration_generation_;
        history_.clear();
        changed_.notify_all();
    }
}

void DepthStore::set_geometry_valid(bool valid) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (geometry_valid_ == valid) return;
    geometry_valid_ = valid;
    ++calibration_generation_;
    history_.clear();
    changed_.notify_all();
}

WorkToken DepthStore::token() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {epoch_, calibration_generation_};
}

bool DepthStore::discard_if_current(WorkToken token) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (token.epoch != epoch_ || token.calibration_generation != calibration_generation_) return false;
    history_.clear();
    changed_.notify_all();
    return true;
}

bool DepthStore::publish(WorkToken token, DepthFrame frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_ || calibrating_ || !geometry_valid_ || token.epoch != epoch_ ||
        token.calibration_generation != calibration_generation_ || frame.epoch != epoch_ ||
        (!history_.empty() && frame.sequence <= history_.back()->sequence)) return false;
    history_.push_back(std::make_shared<const DepthFrame>(std::move(frame)));
    if (history_.size() > capacity_) history_.pop_front();
    changed_.notify_all();
    return true;
}

bool DepthStore::fresh_locked(const DepthFrame& frame, std::chrono::nanoseconds max_age) const {
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    // Subtract only after ordering and sign checks to avoid signed overflow.
    return frame.publish_ns >= 0 && frame.publish_ns <= now && max_age.count() >= 0 &&
           static_cast<std::uint64_t>(now) - static_cast<std::uint64_t>(frame.publish_ns) <=
               static_cast<std::uint64_t>(max_age.count());
}

DepthResult DepthStore::read(std::optional<std::uint64_t> sequence,
                             std::chrono::nanoseconds timeout,
                             std::chrono::nanoseconds max_age) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (calibrating_) return {DepthStatus::calibrating, {}};
    if (!running_ || !geometry_valid_) return {DepthStatus::unavailable, {}};
    const auto request_epoch = epoch_;
    const auto request_calibration = calibration_generation_;
    auto ready = [&] {
        if (!running_ || !geometry_valid_ || calibrating_ || epoch_ != request_epoch ||
            calibration_generation_ != request_calibration) return true;
        if (history_.empty()) return false;
        if (!sequence) return fresh_locked(*history_.back(), max_age);
        return history_.back()->sequence >= *sequence;
    };
    changed_.wait_for(lock, timeout, ready);
    if (calibrating_) return {DepthStatus::calibrating, {}};
    if (!running_ || epoch_ != request_epoch || calibration_generation_ != request_calibration)
        return {DepthStatus::changed, {}};
    if (!geometry_valid_) return {DepthStatus::unavailable, {}};
    if (!sequence) {
        if (history_.empty()) return {DepthStatus::unavailable, {}};
        if (!fresh_locked(*history_.back(), max_age)) return {DepthStatus::warming, {}};
        return {DepthStatus::ok, history_.back()};
    }
    const auto it = std::find_if(history_.rbegin(), history_.rend(),
        [sequence](const auto& frame) { return frame->sequence == *sequence; });
    if (it == history_.rend()) return {DepthStatus::expired, {}};
    return {DepthStatus::ok, *it};
}

} // namespace agtc
