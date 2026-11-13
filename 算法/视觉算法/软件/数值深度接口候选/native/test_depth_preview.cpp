#include "depth_preview.h"
#include <turbojpeg.h>
#include <opencv2/imgproc.hpp>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cstdlib>

using namespace std::chrono_literals;
int main() {
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    store.set_capture_running(true);
    auto token = store.token();
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    agtc::DepthPreview preview(store);
    assert(preview.latest(0ns, 1s).status == agtc::DepthStatus::unavailable);
    std::vector<std::uint8_t> payload(131 * 240 * 2, 0);
    assert(store.publish(token, {1, token.epoch, now, payload, "{}"}));
    auto black = preview.latest(0ns, 1s);
    auto repeated = preview.latest(0ns, 1s);
    assert(black.status == agtc::DepthStatus::ok && black.frame && black.frame == repeated.frame);
    tjhandle decoder = tjInitDecompress();
    assert(decoder);
    int width = 0, height = 0, subsampling = 0, colorspace = 0;
    assert(tjDecompressHeader3(decoder, black.frame->jpeg.data(), static_cast<unsigned long>(black.frame->jpeg.size()),
                               &width, &height, &subsampling, &colorspace) == 0);
    assert(width == 131 && height == 240);
    std::vector<unsigned char> pixels(width * height * 3);
    assert(tjDecompress2(decoder, black.frame->jpeg.data(), static_cast<unsigned long>(black.frame->jpeg.size()),
                         pixels.data(), width, 0, height, TJPF_BGR, 0) == 0);
    for (auto channel : pixels) assert(channel <= 3);
    tjDestroy(decoder);
    payload[0] = 1; // unsupported nonzero millimetres must fail closed
    assert(store.publish(token, {2, token.epoch, now, payload, "{}"}));
    assert(preview.latest(0ns, 1s).status == agtc::DepthStatus::unavailable);
    for (std::size_t i = 0; i < payload.size(); i += 2) { payload[i] = 0x2c; payload[i + 1] = 1; } // 300 mm
    assert(store.publish(token, {3, token.epoch, now, payload, "{}"}));
    auto colored = preview.latest(0ns, 1s);
    assert(colored.status == agtc::DepthStatus::ok && colored.frame != black.frame);
    assert(colored.frame->source->sequence == 3 && colored.frame->jpeg != black.frame->jpeg);
    decoder = tjInitDecompress();
    assert(decoder);
    assert(tjDecompress2(decoder, colored.frame->jpeg.data(), static_cast<unsigned long>(colored.frame->jpeg.size()),
                         pixels.data(), width, 0, height, TJPF_BGR, 0) == 0);
    cv::Mat shade(1, 1, CV_8UC1, cv::Scalar(255)), reference;
    cv::applyColorMap(shade, reference, cv::COLORMAP_TURBO);
    for (int channel = 0; channel < 3; ++channel)
        assert(std::abs(static_cast<int>(pixels[channel]) - reference.at<cv::Vec3b>(0, 0)[channel]) <= 5);
    tjDestroy(decoder);
    store.set_capture_running(false);
    assert(preview.latest(0ns, 1s).status == agtc::DepthStatus::unavailable);
}
