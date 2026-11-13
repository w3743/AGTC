#include "depth_preview.h"
#include <opencv2/imgproc.hpp>
#include <turbojpeg.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace agtc {
DepthPreview::DepthPreview(DepthStore& store) : store_(store), encoder_(tjInitCompress()) {
    if (!encoder_) throw std::runtime_error("JPEG encoder initialization failed");
}
DepthPreview::~DepthPreview() { if (encoder_) tjDestroy(encoder_); }
PreviewResult DepthPreview::latest(std::chrono::nanoseconds timeout, std::chrono::nanoseconds max_age) {
    auto source = store_.read({}, timeout, max_age);
    if (source.status != DepthStatus::ok || !source.frame) return {source.status, {}};
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        auto current = store_.read({}, std::chrono::nanoseconds(0), max_age);
        if (current.status != DepthStatus::ok || current.frame != source.frame) return {DepthStatus::warming, {}};
        if (!cached_ || cached_->source != source.frame) {
            constexpr int width = 131, height = 240;
            if (source.frame->payload.size() != width * height * 2) return {DepthStatus::unavailable, {}};
            cv::Mat shade(height, width, CV_8UC1), mask(height, width, CV_8UC1);
            for (int y = 0; y < height; ++y) {
                auto* s = shade.ptr<std::uint8_t>(y); auto* m = mask.ptr<std::uint8_t>(y);
                for (int x = 0; x < width; ++x) {
                    const auto i = static_cast<std::size_t>(y * width + x) * 2;
                    unsigned mm = source.frame->payload[i] | (static_cast<unsigned>(source.frame->payload[i + 1]) << 8);
                    if (mm != 0 && (mm < 300 || mm > 5000)) return {DepthStatus::unavailable, {}};
                    m[x] = mm ? 255 : 0;
                    s[x] = mm ? static_cast<std::uint8_t>((5000U - mm) * 255U / 4700U) : 0;
                }
            }
            cv::Mat bgr;
            cv::applyColorMap(shade, bgr, cv::COLORMAP_TURBO);
            bgr.setTo(cv::Scalar(0, 0, 0), mask == 0);
            const auto max_size = tjBufSize(width, height, TJSAMP_420);
            if (max_size == std::numeric_limits<unsigned long>::max()) return {DepthStatus::unavailable, {}};
            std::vector<std::uint8_t> bytes(max_size);
            unsigned char* output = bytes.data();
            unsigned long actual = max_size;
            if (tjCompress2(encoder_, bgr.ptr<unsigned char>(), width, static_cast<int>(bgr.step), height,
                            TJPF_BGR, &output, &actual, TJSAMP_420, 85, TJFLAG_NOREALLOC) != 0 ||
                output != bytes.data() || actual > bytes.size()) return {DepthStatus::unavailable, {}};
            bytes.resize(actual);
            cached_ = std::make_shared<const PreviewFrame>(PreviewFrame{source.frame, std::move(bytes)});
        }
        // A pipeline restart, stale source or newer frame while encoding cannot
        // be silently returned as a current preview.
        current = store_.read({}, std::chrono::nanoseconds(0), max_age);
        if (current.status != DepthStatus::ok || current.frame != source.frame) return {DepthStatus::warming, {}};
        return {DepthStatus::ok, cached_};
    } catch (...) { return {DepthStatus::unavailable, {}}; }
}
} // namespace agtc
