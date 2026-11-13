#pragma once
#include "depth_request.h"
#include <string_view>

namespace agtc {
struct HttpDepthResponse {
    int status;
    std::string headers;
    std::string error_body;
    std::shared_ptr<const DepthFrame> frame;
};
// Handles an origin-form GET target; the embedding HTTP server must validate
// the method, HTTP framing and request size and send headers and body in order.
HttpDepthResponse route_depth_get(DepthStore& store, std::string_view target,
                                 std::chrono::nanoseconds timeout,
                                 std::chrono::nanoseconds max_age,
                                 const std::function<void()>& on_valid_request = {});
} // namespace agtc
