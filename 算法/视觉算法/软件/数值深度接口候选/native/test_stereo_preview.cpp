#include "stereo_preview.h"
#include <turbojpeg.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdlib>

int main() {
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    agtc::StereoPairQueue queue(store);
    auto token = queue.start();
    assert(token);
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    cv::Mat left(360, 640, CV_8UC1, cv::Scalar(42)), right(360, 640, CV_8UC1, cv::Scalar(99));
    assert(queue.push(*token, 0, left, 1'000'000'000, now));
    assert(queue.push(*token, 1, right, 1'030'000'000, now)); // display may use unsynced pairs
    assert(!queue.select(now));
    agtc::StereoPreview preview([&] {
        auto current = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        return queue.select(current, false);
    });
    auto first = preview.latest();
    auto repeated = preview.latest();
    assert(first.frame && repeated.frame == first.frame);
    assert(first.frame->right_pts_ns - first.frame->left_pts_ns == 30'000'000);
    tjhandle decoder = tjInitDecompress();
    assert(decoder);
    int width = 0, height = 0, subsampling = 0, colorspace = 0;
    assert(tjDecompressHeader3(decoder, first.frame->jpeg.data(), static_cast<unsigned long>(first.frame->jpeg.size()),
                               &width, &height, &subsampling, &colorspace) == 0);
    assert(width == 1280 && height == 360);
    std::vector<unsigned char> pixels(width * height);
    assert(tjDecompress2(decoder, first.frame->jpeg.data(), static_cast<unsigned long>(first.frame->jpeg.size()),
                         pixels.data(), width, 0, height, TJPF_GRAY, 0) == 0);
    assert(std::abs(static_cast<int>(pixels[180 * width + 320]) - 42) <= 2);
    assert(std::abs(static_cast<int>(pixels[180 * width + 960]) - 99) <= 2);
    tjDestroy(decoder);
    auto captured = queue.select(now, false);
    int calls = 0;
    agtc::StereoPreview invalidated([&]() -> std::optional<agtc::StereoPair> {
        return ++calls == 1 ? captured : std::nullopt;
    });
    assert(!invalidated.latest().frame);
    calls = 0;
    agtc::StereoPreview invalidated_during_encode([&]() -> std::optional<agtc::StereoPair> {
        return ++calls <= 2 ? captured : std::nullopt;
    });
    assert(!invalidated_during_encode.latest().frame);
    queue.stop();
    assert(!preview.latest().frame);
    auto next_token = queue.start();
    assert(next_token);
    now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    cv::Mat left_4k(2160, 3840, CV_8UC1, cv::Scalar(10)), right_4k(2160, 3840, CV_8UC1, cv::Scalar(200));
    assert(queue.push(*next_token, 0, left_4k, 1, now));
    assert(queue.push(*next_token, 1, right_4k, 1, now));
    auto resized = preview.latest();
    assert(resized.frame && resized.frame != first.frame);
    decoder = tjInitDecompress();
    assert(decoder);
    assert(tjDecompressHeader3(decoder, resized.frame->jpeg.data(), static_cast<unsigned long>(resized.frame->jpeg.size()),
                               &width, &height, &subsampling, &colorspace) == 0);
    assert(width == 1280 && height == 360);
    assert(tjDecompress2(decoder, resized.frame->jpeg.data(), static_cast<unsigned long>(resized.frame->jpeg.size()),
                         pixels.data(), width, 0, height, TJPF_GRAY, 0) == 0);
    assert(std::abs(static_cast<int>(pixels[180 * width + 320]) - 10) <= 2);
    assert(std::abs(static_cast<int>(pixels[180 * width + 960]) - 200) <= 2);
    tjDestroy(decoder);
    queue.stop();
}
