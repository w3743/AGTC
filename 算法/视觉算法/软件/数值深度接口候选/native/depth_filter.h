#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// Contiguous row-major float32 disparity and uint8 support masks; nonzero mask
// bytes indicate valid support. All buffers contain width*height elements and
// must not overlap. Invalid depth is NaN and valid output is 0 or 1.
// `disparity_offset` and `depth_factor` use the calibration's double values.
// Returns 0 on success; -1 on invalid arguments or allocation failure.
int agtc_metric_depth_f32(const float* disparity, const uint8_t* left_support,
                          const uint8_t* right_support, size_t width, size_t height,
                          double disparity_offset, double depth_factor,
                          float* depth, uint8_t* valid);
// Returns the scratch byte count, or zero for invalid/overflowing dimensions.
size_t agtc_metric_depth_workspace_size(size_t width, size_t height);
// Same output as agtc_metric_depth_f32, without allocation; workspace must be
// at least the reported size, exclusive to this call, and disjoint from all
// input/output buffers. On invalid arguments no output buffer is modified.
int agtc_metric_depth_f32_with_workspace(const float* disparity, const uint8_t* left_support,
                          const uint8_t* right_support, size_t width, size_t height,
                          double disparity_offset, double depth_factor,
                          float* depth, uint8_t* valid, uint8_t* workspace, size_t workspace_size);
#ifdef __cplusplus
}
#endif
