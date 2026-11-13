#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Contiguous row-major axis-Z metres -> depth-z-u16le-v1 bytes.
 * The caller owns buffers; output_capacity must be at least count * 2.
 * Returns 0 on success, -1 for invalid arguments. Buffers must not overlap.
 */
int agtc_depth_encode_f32(const float *depth, size_t count, uint8_t *out, size_t output_capacity);
int agtc_depth_encode_f64(const double *depth, size_t count, uint8_t *out, size_t output_capacity);

#ifdef __cplusplus
}
#endif
