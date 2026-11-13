#include "depth_producer.h"
#include "depth_encode.h"
#include "depth_filter.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <utility>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "depth metadata validation requires IEEE finite and NaN semantics"
#endif

namespace agtc {
namespace {
bool text_ok(const std::string& text, bool nonempty = false) {
    if (nonempty && text.empty()) return false;
    for (unsigned char c : text) if (c < 0x20 || c > 0x7e) return false;
    return true;
}
bool text_ok(const std::optional<std::string>& text, bool nonempty = false) {
    return !text || text_ok(*text, nonempty);
}
bool ns_ok(const std::optional<std::int64_t>& value) { return !value || *value >= 0; }
void quote(std::ostream& out, const std::string& value) {
    out << '"';
    for (char c : value) { if (c == '"' || c == '\\') out << '\\'; out << c; }
    out << '"';
}
void text(std::ostream& out, const std::optional<std::string>& value) {
    if (value) quote(out, *value); else out << "null";
}
void number(std::ostream& out, const std::optional<std::int64_t>& value) {
    if (value) out << *value; else out << "null";
}
bool matrix_ok(const std::optional<std::array<double, 9>>& K) {
    if (!K) return true;
    for (auto v : *K) if (!std::isfinite(v)) return false;
    return (*K)[0] > 0 && (*K)[4] > 0 && std::abs((*K)[6]) <= 1e-9 &&
           std::abs((*K)[7]) <= 1e-9 && std::abs((*K)[8] - 1) <= 1e-9;
}
} // namespace

bool publish_depth_f32(DepthStore& store, WorkToken token, std::uint64_t sequence,
                       std::int64_t publish_ns, const float* depth, std::size_t count,
                       const DepthContext& c) {
    constexpr std::size_t width = 131, height = 240;
    if (sequence == 0 || sequence > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        publish_ns < 0 || !depth || count != width * height || !text_ok(c.session_id, true) ||
        !text_ok(c.monotonic_clock_id, true) || !text_ok(c.calibration_id) ||
        !text_ok(c.extrinsics_id) || !text_ok(c.preprocessing_id) || !matrix_ok(c.K) ||
        !ns_ok(c.left_pts_ns) || !ns_ok(c.right_pts_ns) || !ns_ok(c.left_arrival_mono_ns) ||
        !ns_ok(c.right_arrival_mono_ns)) return false;
    std::vector<std::uint8_t> payload(count * 2);
    if (agtc_depth_encode_f32(depth, count, payload.data(), payload.size()) != 0) return false;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"schema\":\"depth-z-u16le-v1\",\"session_id\":"; quote(out, c.session_id);
    out << ",\"pipeline_epoch\":" << token.epoch << ",\"seq\":" << sequence
        << ",\"width\":131,\"height\":240,\"encoding\":\"16UC1\",\"byte_order\":\"little\""
           ",\"unit\":\"mm\",\"convention\":\"camera_z\",\"frame_id\":\"left_rectified_optical\",\"K\":";
    if (!c.K) out << "null";
    else { out << '['; for (std::size_t i = 0; i < c.K->size(); ++i) { if (i) out << ','; out << (*c.K)[i]; } out << ']'; }
    out << ",\"calibration_id\":"; text(out, c.calibration_id);
    out << ",\"extrinsics_id\":"; text(out, c.extrinsics_id);
    out << ",\"preprocessing_id\":"; text(out, c.preprocessing_id);
    out << ",\"min_mm\":300,\"max_mm\":5000,\"left_pts_ns\":"; number(out, c.left_pts_ns);
    out << ",\"right_pts_ns\":"; number(out, c.right_pts_ns);
    out << ",\"left_arrival_mono_ns\":"; number(out, c.left_arrival_mono_ns);
    out << ",\"right_arrival_mono_ns\":"; number(out, c.right_arrival_mono_ns);
    out << ",\"publish_mono_ns\":" << publish_ns
        << ",\"source_mono_ns\":null,\"clock_mapping_id\":null,\"clock_quality\":null,\"monotonic_clock_id\":";
    text(out, c.monotonic_clock_id);
    out << ",\"source_time_reference\":\"left_pts\"}";
    return store.publish(token, DepthFrame{sequence, token.epoch, publish_ns, std::move(payload), out.str()});
}

bool publish_disparity_f32(DepthStore& store, WorkToken token, std::uint64_t sequence,
                           std::int64_t publish_ns, const float* disparity,
                           const std::uint8_t* left_support, const std::uint8_t* right_support,
                           std::size_t count, double offset, double factor, const DepthContext& context) {
    constexpr std::size_t width = 131, height = 240;
    if (count != width * height || !disparity || !left_support || !right_support) return false;
    std::vector<float> depth(count);
    std::vector<std::uint8_t> valid(count);
    if (agtc_metric_depth_f32(disparity, left_support, right_support, width, height,
                              offset, factor, depth.data(), valid.data()) != 0) return false;
    return publish_depth_f32(store, token, sequence, publish_ns, depth.data(), count, context);
}
} // namespace agtc
