#include "depth_request.h"
#include <limits>
#include <utility>

namespace agtc {
DepthResponse handle_depth_request(DepthStore& store,
    const std::optional<std::vector<std::string>>& values,
    std::chrono::nanoseconds timeout, std::chrono::nanoseconds max_age,
    const std::function<void()>& on_valid_request) {
    std::optional<std::uint64_t> sequence;
    if (values) {
        if (values->size() != 1 || values->front().empty() || values->front().size() > 19)
            return {400, DepthStatus::unavailable, {}};
        std::uint64_t n = 0;
        constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        for (unsigned char c : values->front()) {
            if (c < '0' || c > '9' || n > (max - (c - '0')) / 10)
                return {400, DepthStatus::unavailable, {}};
            n = n * 10 + (c - '0');
        }
        if (n == 0) return {400, DepthStatus::unavailable, {}};
        sequence = n;
    }
    if (on_valid_request) on_valid_request();
    auto result = store.read(sequence, timeout, max_age);
    int status = 503;
    switch (result.status) {
        case DepthStatus::ok: status = 200; break;
        case DepthStatus::calibrating: case DepthStatus::expired: status = 409; break;
        case DepthStatus::changed: status = sequence ? 409 : 503; break;
        case DepthStatus::unavailable: status = 503; break;
        case DepthStatus::warming: status = 503; break;
    }
    return {status, result.status, std::move(result.frame)};
}
} // namespace agtc
