// Synthetic appsrc fixture for local HTTP integration only.
#include "depth_http_server.h"
#include "vision_runtime.h"
#ifdef AGTC_WITH_PREVIEW
#include "depth_preview.h"
#include "stereo_preview.h"
#endif
#include <gst/app/gstappsrc.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

struct JoinProducer {
    std::atomic<bool>& active;
    std::thread& thread;
    ~JoinProducer() { active = false; if (thread.joinable()) thread.join(); }
};
static bool push(GstAppSrc* source, std::uint8_t value, GstClockTime pts) {
    constexpr std::size_t y_bytes = 640 * 360;
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, y_bytes * 3 / 2, nullptr);
    if (!buffer) return false;
    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) { gst_buffer_unref(buffer); return false; }
    std::memset(map.data, value, y_bytes);
    std::memset(map.data + y_bytes, 128, y_bytes / 2);
    gst_buffer_unmap(buffer, &map);
    GST_BUFFER_PTS(buffer) = pts;
    GST_BUFFER_DURATION(buffer) = GST_SECOND / 30;
    return gst_app_src_push_buffer(source, buffer) == GST_FLOW_OK;
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    gst_init(&argc, &argv);
    cv::setNumThreads(2);
    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(
        "appsrc name=source0 is-live=true format=time caps=video/x-raw,format=NV12,width=640,height=360,framerate=30/1 "
        "! appsink name=sink0 sync=false max-buffers=2 drop=true "
        "appsrc name=source1 is-live=true format=time caps=video/x-raw,format=NV12,width=640,height=360,framerate=30/1 "
        "! appsink name=sink1 sync=false max-buffers=2 drop=true", &error);
    if (!pipeline || error) { if (error) g_error_free(error); if (pipeline) gst_object_unref(pipeline); return 3; }
    GstElement* left = gst_bin_get_by_name(GST_BIN(pipeline), "source0");
    GstElement* right = gst_bin_get_by_name(GST_BIN(pipeline), "source1");
    if (!left || !right) return 4;
    try {
        agtc::VisionRuntime runtime(argv[1], "service-fixture", "fixture-steady-clock");
        if (!runtime.start(pipeline)) return 5;
        std::atomic<bool> producing{true};
        std::thread producer([&] {
            GstClockTime pts = GST_SECOND;
            while (producing && runtime.running()) {
                if (!push(GST_APP_SRC(left), 42, pts) || !push(GST_APP_SRC(right), 99, pts + 10 * GST_MSECOND)) break;
                pts += GST_SECOND / 30;
                std::this_thread::sleep_for(std::chrono::milliseconds(33));
            }
        });
        JoinProducer join_producer{producing, producer};
#ifdef AGTC_WITH_PREVIEW
        agtc::DepthPreview preview(runtime.store());
        agtc::StereoPreview stereo_preview([&] { return runtime.preview_pair(); });
        agtc::DepthHttpServer http(runtime.store(), [&] { runtime.request_depth(); }, [&] {
            return preview.latest(std::chrono::seconds(1), std::chrono::milliseconds(1500));
        }, [&] { return stereo_preview.latest(); });
#else
        agtc::DepthHttpServer http(runtime.store(), [&] { runtime.request_depth(); });
#endif
        int port = http.start();
        if (port <= 0) { producing = false; producer.join(); runtime.stop(); return 6; }
        std::cout << port << std::endl;
        std::string command;
        while (std::getline(std::cin, command)) {
            if (command == "state") {
                auto latest = runtime.store().read({}, std::chrono::milliseconds(0), std::chrono::hours(1));
                if (latest.frame) std::cout << latest.frame->sequence << std::endl;
                else std::cout << "empty" << std::endl;
            } else if (command == "pause") {
                producing = false;
                if (producer.joinable()) producer.join();
                std::cout << "paused" << std::endl;
            } else if (command == "fault") {
                GError* fault = g_error_new_literal(g_quark_from_static_string("agtc-test"), 1, "injected");
                gst_element_post_message(pipeline, gst_message_new_error(GST_OBJECT(pipeline), fault, "fixture"));
                g_error_free(fault);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                while (!runtime.failed() && std::chrono::steady_clock::now() < until)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                std::cout << (runtime.failed() ? "faulted" : "timeout") << std::endl;
            } else if (command == "stop") break;
        }
        producing = false;
        if (producer.joinable()) producer.join();
        http.stop();
        runtime.stop();
    } catch (const std::exception& exc) { std::cerr << exc.what() << '\n'; return 7; }
    gst_object_unref(left); gst_object_unref(right); gst_object_unref(pipeline);
}
