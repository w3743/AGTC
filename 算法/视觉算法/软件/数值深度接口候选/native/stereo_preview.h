#pragma once
#include "preview_types.h"
#include "stereo_pair_queue.h"
#include <functional>

namespace agtc {
// Display-only unrectified stereo preview. Pair timestamps are not proof of
// hardware exposure synchronization. The provider must retain capture lifetime.
class StereoPreview {
public:
    explicit StereoPreview(std::function<std::optional<StereoPair>()> provider);
    ~StereoPreview();
    StereoPreview(const StereoPreview&) = delete;
    StereoPreview& operator=(const StereoPreview&) = delete;
    StereoPreviewResult latest();
private:
    bool still_current(const StereoPair& pair);
    std::function<std::optional<StereoPair>()> provider_;
    std::mutex mutex_;
    void* encoder_ = nullptr;
    cv::Mat canvas_;
    std::vector<std::uint8_t> jpeg_scratch_;
    std::optional<StereoPair> cached_pair_;
    std::shared_ptr<const StereoPreviewFrame> cached_;
};
} // namespace agtc
