#include "stereo_preview.h"
#include <opencv2/imgproc.hpp>
#include <turbojpeg.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace agtc {
namespace {
bool same_token(WorkToken a, WorkToken b) {
    return a.epoch == b.epoch && a.calibration_generation == b.calibration_generation;
}
bool same_frame(const TimedStereoFrame& a, const TimedStereoFrame& b) {
    return a.pts_ns == b.pts_ns && a.arrival_ns == b.arrival_ns && a.gray.data == b.gray.data;
}
bool fresh(std::int64_t now, std::int64_t arrival) {
    return now >= 0 && arrival >= 0 && now >= arrival && now - arrival <= 750'000'000;
}
} // namespace
StereoPreview::StereoPreview(std::function<std::optional<StereoPair>()> provider)
    : provider_(std::move(provider)), encoder_(tjInitCompress()) {
    if (!provider_ || !encoder_) {
        if (encoder_) tjDestroy(encoder_);
        throw std::invalid_argument("preview provider and JPEG encoder required");
    }
}
StereoPreview::~StereoPreview() { if (encoder_) tjDestroy(encoder_); }
bool StereoPreview::still_current(const StereoPair& pair) {
    auto current = provider_();
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return current && same_token(current->token, pair.token) &&
           fresh(now, pair.left.arrival_ns) && fresh(now, pair.right.arrival_ns);
}
StereoPreviewResult StereoPreview::latest() {
    try {
        auto pair = provider_();
        if (!pair) return {};
        std::lock_guard<std::mutex> lock(mutex_);
        if (!still_current(*pair)) return {};
        if (!cached_ || !cached_pair_ || !same_token(pair->token, cached_pair_->token) ||
            !same_frame(pair->left, cached_pair_->left) || !same_frame(pair->right, cached_pair_->right)) {
            if (pair->left.gray.type() != CV_8UC1 || pair->right.gray.type() != CV_8UC1 ||
                pair->left.gray.empty() || pair->right.gray.empty() || pair->left.gray.size() != pair->right.gray.size() ||
                (pair->left.gray.size() != cv::Size(640, 360) && pair->left.gray.size() != cv::Size(3840, 2160))) return {};
            canvas_.create(360, 1280, CV_8UC1);
            cv::Mat left = canvas_(cv::Rect(0, 0, 640, 360));
            cv::Mat right = canvas_(cv::Rect(640, 0, 640, 360));
            if (pair->left.gray.size() == cv::Size(640, 360)) pair->left.gray.copyTo(left);
            else cv::resize(pair->left.gray, left, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
            if (pair->right.gray.size() == cv::Size(640, 360)) pair->right.gray.copyTo(right);
            else cv::resize(pair->right.gray, right, cv::Size(640, 360), 0, 0, cv::INTER_LINEAR);
            const auto capacity = tjBufSize(1280, 360, TJSAMP_GRAY);
            if (capacity == std::numeric_limits<unsigned long>::max()) return {};
            jpeg_scratch_.resize(capacity);
            unsigned char* output = jpeg_scratch_.data();
            unsigned long actual = capacity;
            if (tjCompress2(encoder_, canvas_.ptr<unsigned char>(), 1280, static_cast<int>(canvas_.step), 360,
                            TJPF_GRAY, &output, &actual, TJSAMP_GRAY, 55, TJFLAG_NOREALLOC) != 0 ||
                output != jpeg_scratch_.data() || actual > jpeg_scratch_.size()) return {};
            std::vector<std::uint8_t> bytes(jpeg_scratch_.begin(), jpeg_scratch_.begin() + actual);
            auto next = std::make_shared<const StereoPreviewFrame>(StereoPreviewFrame{
                pair->token, pair->left.pts_ns, pair->right.pts_ns, std::move(bytes)});
            cached_pair_ = *pair;
            cached_ = std::move(next);
        }
        if (!still_current(*pair)) return {};
        return {cached_};
    } catch (...) { return {}; }
}
} // namespace agtc
