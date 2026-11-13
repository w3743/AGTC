#include "gst_stereo_capture.h"
#include "depth_http.h"
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstring>
#include <fstream>
#include <future>

using namespace std::chrono_literals;
static GstElement* make_pipeline() {
    GError* error = nullptr;
    GstElement* p = gst_parse_launch(
        "appsrc name=source0 is-live=true format=time caps=video/x-raw,format=NV12,width=640,height=360,framerate=30/1 "
        "! appsink name=sink0 sync=false max-buffers=2 drop=true "
        "appsrc name=source1 is-live=true format=time caps=video/x-raw,format=NV12,width=640,height=360,framerate=30/1 "
        "! appsink name=sink1 sync=false max-buffers=2 drop=true", &error);
    assert(!error && p && GST_IS_PIPELINE(p));
    return p;
}
static void push(GstElement* pipeline, const char* name, std::uint8_t value, GstClockTime pts) {
    constexpr int width = 640, height = 360, stride = 672;
    const gsize bytes = stride * height * 3 / 2;
    GstBuffer* b = gst_buffer_new_allocate(nullptr, bytes, nullptr);
    assert(b);
    GstMapInfo map{};
    assert(gst_buffer_map(b, &map, GST_MAP_WRITE));
    std::memset(map.data, value, stride * height);
    std::memset(map.data + stride * height, 128, stride * height / 2);
    gst_buffer_unmap(b, &map);
    gsize offsets[GST_VIDEO_MAX_PLANES] = {0, stride * height, 0, 0};
    gint strides[GST_VIDEO_MAX_PLANES] = {stride, stride, 0, 0};
    assert(gst_buffer_add_video_meta_full(b, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_NV12,
                                           width, height, 2, offsets, strides));
    GST_BUFFER_PTS(b) = pts;
    GST_BUFFER_DURATION(b) = GST_SECOND / 30;
    GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), name);
    assert(source && GST_IS_APP_SRC(source));
    auto flow = gst_app_src_push_buffer(GST_APP_SRC(source), b);
    gst_object_unref(source);
    assert(flow == GST_FLOW_OK);
}
int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    assert(!agtc::make_v4l2_stereo_pipeline("", "/dev/video1"));
    assert(!agtc::make_v4l2_stereo_pipeline("/dev/video0", "/dev/video0"));
    assert(!agtc::make_v4l2_stereo_pipeline(std::string("/dev/video0\0x", 13), "/dev/video1"));
    assert(!agtc::make_v4l2_stereo_pipeline("/dev/video0", "/dev/video1", 320, 240));
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    agtc::GstStereoCapture capture(store);
    GstElement* invalid = gst_pipeline_new(nullptr);
    assert(!capture.start(invalid));
    gst_object_unref(invalid);
    GstElement* pipeline = make_pipeline();
    assert(capture.start(pipeline));
    assert(!capture.start(pipeline));
    push(pipeline, "source0", 42, GST_SECOND);
    push(pipeline, "source1", 99, GST_SECOND + 10 * GST_MSECOND);
    assert(capture.poll(0, 1000ms));
    assert(capture.poll(1, 1000ms));
    auto pair = capture.select();
    assert(pair && pair->left.pts_ns == static_cast<std::int64_t>(GST_SECOND));
    assert(pair->right.pts_ns == static_cast<std::int64_t>(GST_SECOND + 10 * GST_MSECOND));
    assert(pair->left.gray.isContinuous() && pair->left.gray.at<std::uint8_t>(25, 40) == 42);
    assert(pair->right.gray.at<std::uint8_t>(25, 40) == 99);
    auto old_token = pair->token;
    if (argc == 2 || argc == 3) {
        agtc::StereoSgbm stereo(argv[1]);
        auto clock = [] { return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count(); };
        agtc::StereoPublisher publisher(stereo, store, "gst-test", "test-steady-clock", clock);
        assert(publisher.process(pair->token, pair->left, pair->right) == agtc::PublishStatus::ok);
        auto response = agtc::route_depth_get(store, "/depth.bin", 0ms, 1s);
        assert(response.status == 200 && response.frame && response.frame->payload.size() == 131 * 240 * 2);
        if (argc == 3) {
            std::ofstream file(argv[2], std::ios::binary);
            file.write(response.headers.data(), static_cast<std::streamsize>(response.headers.size()));
            file.write(reinterpret_cast<const char*>(response.frame->payload.data()),
                       static_cast<std::streamsize>(response.frame->payload.size()));
            assert(file.good());
        }
    }
    GError* error = g_error_new_literal(g_quark_from_static_string("agtc-test"), 1, "injected failure");
    gst_element_post_message(pipeline, gst_message_new_error(GST_OBJECT(pipeline), error, "fixture"));
    g_error_free(error);
    assert(!capture.select());
    assert(!capture.poll(0, 0ms));
    assert(store.read({}, 0ms, 1s).status != agtc::DepthStatus::ok);
    assert(pair->left.gray.at<std::uint8_t>(25, 40) == 42); // released GstBuffer remains independent
    gst_object_unref(pipeline);
    pipeline = make_pipeline();
    assert(capture.start(pipeline));
    push(pipeline, "source0", 7, 0);
    push(pipeline, "source1", 8, 0);
    assert(capture.poll(0, 1000ms) && capture.poll(1, 1000ms));
    pair = capture.select();
    assert(pair && pair->token.epoch != old_token.epoch && pair->left.gray.at<std::uint8_t>(0, 0) == 7);
    GstElement* source = gst_bin_get_by_name(GST_BIN(pipeline), "source0");
    assert(source && gst_app_src_end_of_stream(GST_APP_SRC(source)) == GST_FLOW_OK);
    gst_object_unref(source);
    assert(!capture.poll(0, 1000ms));
    assert(!capture.select());
    capture.stop();
    gst_object_unref(pipeline);
    pipeline = make_pipeline();
    assert(capture.start(pipeline));
    push(pipeline, "source0", 7, GST_SECOND);
    push(pipeline, "source1", 8, GST_SECOND);
    assert(capture.poll(0, 1000ms) && capture.poll(1, 1000ms));
    push(pipeline, "source0", 7, 0);
    assert(!capture.poll(0, 1000ms)); // PTS rollback invalidates the session.
    assert(!capture.select());
    gst_object_unref(pipeline);
    pipeline = make_pipeline();
    assert(capture.start(pipeline));
    auto pending = std::async(std::launch::async, [&] { return capture.poll(0, 1000ms); });
    assert(pending.wait_for(50ms) == std::future_status::timeout);
    capture.stop();
    assert(!pending.get());
    gst_object_unref(pipeline);
}
