#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace agtc {

// Immutable record after publication; metadata is already validated/serialized
// by the producer. Timestamps must use the same monotonic clock as read().
struct DepthFrame {
    std::uint64_t sequence;
    std::uint64_t epoch;
    std::int64_t publish_ns;
    std::vector<std::uint8_t> payload;
    std::string metadata_ascii;
};

struct WorkToken { std::uint64_t epoch; std::uint64_t calibration_generation; };
enum class DepthStatus { ok, calibrating, unavailable, changed, warming, expired };
struct DepthResult { DepthStatus status; std::shared_ptr<const DepthFrame> frame; };

// Capture/calibration owners must set state before touching the hardware. A
// token is acquired before work starts and checked atomically at publish.
class DepthStore {
public:
    explicit DepthStore(std::size_t capacity);
    void set_capture_running(bool running);
    void set_calibrating(bool active);
    void set_geometry_valid(bool valid);
    WorkToken token() const;
    bool publish(WorkToken token, DepthFrame frame);
    // Invalidate cached frames only if the producer still owns this generation.
    bool discard_if_current(WorkToken token);
    DepthResult read(std::optional<std::uint64_t> sequence,
                     std::chrono::nanoseconds timeout,
                     std::chrono::nanoseconds max_age);
private:
    bool fresh_locked(const DepthFrame& frame, std::chrono::nanoseconds max_age) const;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    const std::size_t capacity_;
    std::uint64_t epoch_ = 0, calibration_generation_ = 0;
    bool running_ = false, calibrating_ = false, geometry_valid_ = false;
    std::deque<std::shared_ptr<const DepthFrame>> history_;
};

} // namespace agtc
