// Incomplete loopback-only candidate. Not a replacement for the deployed UI.
#include "depth_http_server.h"
#include "vision_runtime.h"
#ifdef AGTC_WITH_PREVIEW
#include "depth_preview.h"
#include "stereo_preview.h"
#endif
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

namespace { volatile std::sig_atomic_t stopping = 0; void on_signal(int) { stopping = 1; } }
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: vision_candidate_main calibration.yml left-device right-device session-id clock-id port\n";
        return 2;
    }
    try {
        int port = std::stoi(argv[6]);
        if (port < 1 || port > 65535) throw std::invalid_argument("invalid port");
        gst_init(nullptr, nullptr);
        cv::setNumThreads(2); // Match the existing fast-SGBM profile until target-board tuning.
        agtc::VisionRuntime runtime(argv[1], argv[4], argv[5]);
        std::unique_ptr<GstElement, void(*)(GstElement*)> pipeline(
            agtc::make_v4l2_stereo_pipeline(argv[2], argv[3]),
            [](GstElement* element) { if (element) gst_object_unref(element); });
        if (!pipeline) { std::cerr << "could not create V4L2 pipeline\n"; return 1; }
        bool started = runtime.start(pipeline.get());
        if (!started) { std::cerr << "capture startup failed\n"; return 1; }
#ifdef AGTC_WITH_PREVIEW
        agtc::DepthPreview preview(runtime.store());
        agtc::StereoPreview stereo_preview([&] { return runtime.preview_pair(); });
        agtc::DepthHttpServer http(runtime.store(), [&] { runtime.request_depth(); }, [&] {
            return preview.latest(std::chrono::seconds(1), std::chrono::milliseconds(1500));
        }, [&] { return stereo_preview.latest(); });
#else
        agtc::DepthHttpServer http(runtime.store(), [&] { runtime.request_depth(); });
#endif
        if (http.start(port) < 0) { runtime.stop(); std::cerr << "HTTP bind failed\n"; return 1; }
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::cout << "candidate depth endpoint listening on 127.0.0.1:" << port << std::endl;
        while (!stopping && runtime.running()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        http.stop();
        runtime.stop();
        if (runtime.failed()) { std::cerr << "capture or processing failed\n"; return 1; }
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
