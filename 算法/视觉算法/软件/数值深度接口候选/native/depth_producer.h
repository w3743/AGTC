#pragma once
#include "depth_store.h"
#include <array>

namespace agtc {
// This subset of the wire metadata explicitly identifies unverified geometry
// with null values. Strings are printable ASCII; the caller must not invent
// calibration, extrinsic, clock-mapping or timing evidence.
struct DepthContext {
    std::string session_id;
    std::optional<std::string> monotonic_clock_id;
    std::optional<std::string> calibration_id, extrinsics_id, preprocessing_id;
    std::optional<std::array<double, 9>> K;
    std::optional<std::int64_t> left_pts_ns, right_pts_ns;
    std::optional<std::int64_t> left_arrival_mono_ns, right_arrival_mono_ns;
};

// Makes a 131x240 float32 frame with its metadata and publishes atomically.
// Caller supplies a unique positive monotonically increasing sequence number
// and a timestamp from the same steady_clock domain used by DepthStore.
// False means invalid input or a stale/rejected work token.
bool publish_depth_f32(DepthStore& store, WorkToken token, std::uint64_t sequence,
                       std::int64_t publish_ns, const float* depth, std::size_t count,
                       const DepthContext& context);
// Applies the native metric filter to a fixed-profile float32 disparity raster,
// then encodes and atomically publishes it. Support masks must be 131x240.
bool publish_disparity_f32(DepthStore& store, WorkToken token, std::uint64_t sequence,
                           std::int64_t publish_ns, const float* disparity,
                           const std::uint8_t* left_support, const std::uint8_t* right_support,
                           std::size_t count, double disparity_offset, double depth_factor,
                           const DepthContext& context);
} // namespace agtc
