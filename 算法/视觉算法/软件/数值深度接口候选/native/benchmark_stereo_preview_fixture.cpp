// Host-only synthetic gray-frame JPEG microbenchmark.
#include "stereo_preview.h"
#include <opencv2/core.hpp>
#include <turbojpeg.h>
#include <chrono>
#include <iostream>

int main() {
    cv::setNumThreads(2);
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    agtc::StereoPairQueue queue(store);
    auto token = queue.start();
    if (!token) return 1;
    auto clock = [] { return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };
    cv::Mat left(360, 640, CV_8UC1, cv::Scalar(42)), right(360, 640, CV_8UC1, cv::Scalar(99));
    agtc::StereoPreview preview([&] { return queue.select(clock(), false); });
    for (int i = 0; i < 275; ++i) {
        auto now = clock();
        if (!queue.push(*token, 0, left, i, now) || !queue.push(*token, 1, right, i, now)) return 2;
        auto start = std::chrono::steady_clock::now();
        auto image = preview.latest();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        if (!image.frame) return 3;
        if (i >= 25) std::cout << "fresh " << elapsed << '\n';
    }
    for (int i = 0; i < 275; ++i) {
        auto start = std::chrono::steady_clock::now();
        auto image = preview.latest();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
        if (!image.frame) return 4;
        if (i >= 25) std::cout << "cached " << elapsed << '\n';
    }
    cv::Mat canvas;
    auto encoder = tjInitCompress();
    if (!encoder) return 5;
    const auto capacity = tjBufSize(1280, 360, TJSAMP_GRAY);
    std::vector<unsigned char> jpeg(capacity);
    for (int i = 0; i < 275; ++i) {
        auto start = std::chrono::steady_clock::now();
        cv::hconcat(left, right, canvas);
        auto middle = std::chrono::steady_clock::now();
        unsigned char* output = jpeg.data();
        unsigned long bytes = capacity;
        if (tjCompress2(encoder, canvas.ptr<unsigned char>(), 1280, static_cast<int>(canvas.step), 360,
                        TJPF_GRAY, &output, &bytes, TJSAMP_GRAY, 55, TJFLAG_NOREALLOC) != 0) return 6;
        auto finish = std::chrono::steady_clock::now();
        if (i >= 25) {
            std::cout << "concat " << std::chrono::duration_cast<std::chrono::nanoseconds>(middle - start).count() << '\n';
            std::cout << "jpeg " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - middle).count() << '\n';
        }
    }
    tjDestroy(encoder);
}
