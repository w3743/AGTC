#pragma once
#include "preview_types.h"

namespace agtc {
// Fixed 0.3..5 m preview from the quantized depth wire frame. It is not pixel
// equivalent to the old unquantized/auto-range preview. One encoder is guarded
// by a mutex and JPEGs are cached by immutable source frame identity.
class DepthPreview {
public:
    explicit DepthPreview(DepthStore& store);
    ~DepthPreview();
    DepthPreview(const DepthPreview&) = delete;
    DepthPreview& operator=(const DepthPreview&) = delete;
    PreviewResult latest(std::chrono::nanoseconds timeout, std::chrono::nanoseconds max_age);
private:
    DepthStore& store_;
    std::mutex mutex_;
    void* encoder_ = nullptr;
    std::shared_ptr<const PreviewFrame> cached_;
};
} // namespace agtc
