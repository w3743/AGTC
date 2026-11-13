#pragma once
#include "depth_store.h"

namespace agtc {
struct PreviewFrame {
    std::shared_ptr<const DepthFrame> source;
    std::vector<std::uint8_t> jpeg;
};
struct PreviewResult {
    DepthStatus status;
    std::shared_ptr<const PreviewFrame> frame;
};
struct StereoPreviewFrame {
    WorkToken token;
    std::int64_t left_pts_ns;
    std::int64_t right_pts_ns;
    std::vector<std::uint8_t> jpeg;
};
struct StereoPreviewResult { std::shared_ptr<const StereoPreviewFrame> frame; };
} // namespace agtc
