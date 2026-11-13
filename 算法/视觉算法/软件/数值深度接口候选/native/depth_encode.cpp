#include "depth_encode.h"

#include <cstring>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "depth encoding requires IEEE NaN and ordered-comparison semantics; disable fast-math and finite-math-only"
#endif

namespace {
// Round the already-scaled, nonnegative finite value to nearest, ties to even.
uint16_t round_mm(double value) {
    const auto whole = static_cast<unsigned>(value);
    const double floor_value = static_cast<double>(whole);
    const double fraction = value - floor_value;
    auto result = whole;
    result += (fraction > 0.5) | ((fraction == 0.5) & ((result & 1U) != 0));
    return static_cast<uint16_t>(result);
}

template <typename T>
int encode(const T *depth, size_t count, uint8_t *out, size_t capacity) {
    if (!depth || !out || count == 0 || count > std::numeric_limits<size_t>::max() / 2 || capacity < count * 2) return -1;
    for (size_t i = 0; i < count; ++i) {
        const T value = depth[i];
        // NumPy compares float32 arrays using float32-cast scalar bounds.
        // Ordered comparisons also reject NaN and infinities before conversion.
        const bool valid = (value >= static_cast<T>(0.3)) & (value <= static_cast<T>(5.0));
        // Clear the representation of invalid values before conversion to an
        // integer. This also avoids an unpredictable branch on sparse depth.
        double safe = static_cast<double>(value);
        uint64_t bits;
        static_assert(sizeof(bits) == sizeof(safe) && std::numeric_limits<double>::is_iec559,
                      "requires IEEE 754 binary64 double");
        std::memcpy(&bits, &safe, sizeof(bits));
        bits &= uint64_t(0) - static_cast<uint64_t>(valid);
        std::memcpy(&safe, &bits, sizeof(safe));
        const uint16_t mm = round_mm(safe * 1000.0);
        out[2 * i] = static_cast<uint8_t>(mm & 0xffU);
        out[2 * i + 1] = static_cast<uint8_t>(mm >> 8);
    }
    return 0;
}
}  // namespace

extern "C" int agtc_depth_encode_f32(const float *depth, size_t count, uint8_t *out, size_t capacity) {
    return encode(depth, count, out, capacity);
}
extern "C" int agtc_depth_encode_f64(const double *depth, size_t count, uint8_t *out, size_t capacity) {
    return encode(depth, count, out, capacity);
}
