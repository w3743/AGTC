#pragma once
#include "depth_store.h"
#include <functional>

namespace agtc {
// Query values are already URL-decoded by the HTTP parser. An absent optional
// means no sequence parameter; present with zero/multiple values is invalid.
struct DepthResponse {
    int status;
    DepthStatus reason;
    std::shared_ptr<const DepthFrame> frame;
};
DepthResponse handle_depth_request(DepthStore& store,
    const std::optional<std::vector<std::string>>& sequence_values,
    std::chrono::nanoseconds timeout, std::chrono::nanoseconds max_age,
    const std::function<void()>& on_valid_request = {});
} // namespace agtc
