#include "depth_filter.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "depth filter requires IEEE NaN and ordered-comparison semantics"
#endif

namespace {
bool valid_parameters(double offset, double factor) {
    return std::isfinite(offset) && std::isfinite(factor) && factor > 0 &&
           std::isfinite(static_cast<float>(offset)) && std::isfinite(static_cast<float>(factor)) &&
           std::isfinite(static_cast<float>(offset + .01));
}
}

extern "C" size_t agtc_metric_depth_workspace_size(size_t width, size_t height) {
    const auto max = std::numeric_limits<size_t>::max();
    if (!width || !height || width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        width > max - 2 || height > max / width) return 0;
    const size_t count = width * height;
    if (count > max - (width + 2)) return 0;
    return count + width + 2;
}

extern "C" int agtc_metric_depth_f32_with_workspace(const float* disparity, const uint8_t* left_support,
                                      const uint8_t* right_support, size_t width, size_t height,
                                      double offset, double factor, float* depth, uint8_t* valid,
                                      uint8_t* workspace, size_t workspace_size) {
    const size_t needed = agtc_metric_depth_workspace_size(width, height);
    if (!disparity || !left_support || !right_support || !depth || !valid || !workspace ||
        !needed || workspace_size < needed || !valid_parameters(offset, factor)) return -1;
    const size_t count = width * height;
    std::fill_n(workspace, needed, uint8_t{0});
    uint8_t* candidate = workspace;
    uint8_t* vertical = workspace + count; // each vertical sum is at most three
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float threshold = static_cast<float>(offset + .01);
    const float offset32 = static_cast<float>(offset);
    const float factor32 = static_cast<float>(factor);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            const size_t i = y * width + x;
            const float d = disparity[i];
            depth[i] = nan;
            if (!std::isfinite(d) || !(d > threshold) || !left_support[i]) continue;
            const double xr = static_cast<double>(x) - static_cast<double>(d);
            if (!(xr >= 0 && xr <= static_cast<double>(width - 1))) continue;
            const size_t x0 = static_cast<size_t>(std::floor(xr));
            const size_t x1 = static_cast<size_t>(std::ceil(xr));
            if (!right_support[y * width + x0] || !right_support[y * width + x1]) continue;
            const float denominator = d - offset32;
            if (!std::isfinite(denominator) || denominator <= 0) continue;
            const float z = factor32 / denominator;
            if (!std::isfinite(z) || z < .3f || z > 5.0f) continue;
            depth[i] = z;
            candidate[i] = 1;
        }
    }
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            unsigned sum = candidate[y * width + x];
            if (y > 0) sum += candidate[(y - 1) * width + x];
            if (y + 1 < height) sum += candidate[(y + 1) * width + x];
            vertical[x + 1] = static_cast<uint8_t>(sum);
        }
        for (size_t x = 0; x < width; ++x) {
            const size_t i = y * width + x;
            const bool keep = candidate[i] && vertical[x] + vertical[x + 1] + vertical[x + 2] >= 3;
            valid[i] = static_cast<uint8_t>(keep);
            if (!keep) depth[i] = nan;
        }
    }
    return 0;
}

extern "C" int agtc_metric_depth_f32(const float* disparity, const uint8_t* left_support,
                                      const uint8_t* right_support, size_t width, size_t height,
                                      double offset, double factor, float* depth, uint8_t* valid) {
    const size_t needed = agtc_metric_depth_workspace_size(width, height);
    if (!needed || !disparity || !left_support || !right_support || !depth || !valid || !valid_parameters(offset, factor)) return -1;
    try {
        std::vector<uint8_t> workspace(needed);
        return agtc_metric_depth_f32_with_workspace(disparity, left_support, right_support, width, height,
                                                     offset, factor, depth, valid, workspace.data(), workspace.size());
    } catch (const std::bad_alloc&) { return -1; }
      catch (const std::length_error&) { return -1; }
}
