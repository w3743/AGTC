#include "gst_stereo_capture.h"
#include <gst/video/video.h>
#include <algorithm>
#include <limits>

namespace agtc {
namespace {
bool same(WorkToken a, WorkToken b) {
    return a.epoch == b.epoch && a.calibration_generation == b.calibration_generation;
}
std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

GstElement* make_v4l2_stereo_pipeline(const std::string& left, const std::string& right, int width, int height) {
    if (left.empty() || right.empty() || left == right || left.find('\0') != std::string::npos ||
        right.find('\0') != std::string::npos ||
        !((width == 640 && height == 360) || (width == 3840 && height == 2160))) return nullptr;
    GstElement* pipeline = gst_pipeline_new(nullptr);
    if (!pipeline) return nullptr;
    const std::string* devices[] = {&left, &right};
    for (int i = 0; i < 2; ++i) {
        GstElement* source = gst_element_factory_make("v4l2src", nullptr);
        GstElement* filter = gst_element_factory_make("capsfilter", nullptr);
        GstElement* sink = gst_element_factory_make("appsink", i == 0 ? "sink0" : "sink1");
        if (!source || !filter || !sink) {
            if (source) gst_object_unref(source);
            if (filter) gst_object_unref(filter);
            if (sink) gst_object_unref(sink);
            gst_object_unref(pipeline);
            return nullptr;
        }
        GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "NV12",
            "width", G_TYPE_INT, width, "height", G_TYPE_INT, height,
            "framerate", GST_TYPE_FRACTION, 30, 1, nullptr);
        if (!caps) { gst_object_unref(source); gst_object_unref(filter); gst_object_unref(sink); gst_object_unref(pipeline); return nullptr; }
        g_object_set(source, "device", devices[i]->c_str(), nullptr);
        g_object_set(filter, "caps", caps, nullptr);
        g_object_set(sink, "max-buffers", 2U, "drop", TRUE, "sync", FALSE, nullptr);
        gst_caps_unref(caps);
        gst_bin_add_many(GST_BIN(pipeline), source, filter, sink, nullptr);
        if (!gst_element_link_many(source, filter, sink, nullptr)) { gst_object_unref(pipeline); return nullptr; }
    }
    return pipeline;
}

GstStereoCapture::GstStereoCapture(DepthStore& store) : queue_(store) {}
GstStereoCapture::~GstStereoCapture() { stop(); }
bool GstStereoCapture::start(GstElement* pipeline) {
    std::lock_guard<std::mutex> lock(control_);
    if (pipeline_ || !pipeline || !GST_IS_PIPELINE(pipeline)) return false;
    GstElement* left = gst_bin_get_by_name(GST_BIN(pipeline), "sink0");
    GstElement* right = gst_bin_get_by_name(GST_BIN(pipeline), "sink1");
    if (!left || !right || !GST_IS_APP_SINK(left) || !GST_IS_APP_SINK(right)) {
        if (left) gst_object_unref(left);
        if (right) gst_object_unref(right);
        return false;
    }
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(left); gst_object_unref(right);
        return false;
    }
    auto token = queue_.start();
    if (!token) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(left); gst_object_unref(right);
        return false;
    }
    pipeline_ = GST_ELEMENT(gst_object_ref(pipeline));
    sinks_[0] = GST_APP_SINK(left); sinks_[1] = GST_APP_SINK(right);
    token_ = *token;
    return true;
}
void GstStereoCapture::stop() {
    std::lock_guard<std::mutex> lock(control_);
    if (!pipeline_) return;
    queue_.stop();
    token_.reset();
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(sinks_[0]); gst_object_unref(sinks_[1]);
    gst_object_unref(pipeline_);
    sinks_[0] = sinks_[1] = nullptr;
    pipeline_ = nullptr;
}
void GstStereoCapture::stop_if_current(WorkToken token) {
    std::lock_guard<std::mutex> lock(control_);
    if (!pipeline_ || !token_ || !same(token, *token_)) return;
    queue_.stop();
    token_.reset();
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(sinks_[0]); gst_object_unref(sinks_[1]);
    gst_object_unref(pipeline_);
    sinks_[0] = sinks_[1] = nullptr;
    pipeline_ = nullptr;
}
bool GstStereoCapture::check_bus(WorkToken token, GstElement* pipeline) {
    GstBus* bus = gst_element_get_bus(pipeline);
    if (!bus) { stop_if_current(token); return false; }
    GstMessage* message = gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    gst_object_unref(bus);
    if (!message) return true;
    gst_message_unref(message);
    stop_if_current(token);
    return false;
}
bool GstStereoCapture::poll(int camera, std::chrono::milliseconds timeout) {
    if (camera < 0 || camera > 1 || timeout.count() < 0 || timeout > std::chrono::seconds(1)) return false;
    GstAppSink* sink;
    GstElement* pipeline;
    WorkToken token;
    {
        std::lock_guard<std::mutex> lock(control_);
        if (!pipeline_ || !token_) return false;
        token = *token_;
        sink = GST_APP_SINK(gst_object_ref(sinks_[camera]));
        pipeline = GST_ELEMENT(gst_object_ref(pipeline_));
    }
    if (!check_bus(token, pipeline)) { gst_object_unref(sink); gst_object_unref(pipeline); return false; }
    GstSample* sample = gst_app_sink_try_pull_sample(sink, static_cast<GstClockTime>(timeout.count()) * GST_MSECOND);
    bool accepted = false;
    bool fault = false;
    if (sample) {
        GstVideoInfo info;
        gst_video_info_init(&info);
        GstCaps* caps = gst_sample_get_caps(sample);
        GstBuffer* buffer = gst_sample_get_buffer(sample);
        const GstSegment* segment = gst_sample_get_segment(sample);
        if (caps && buffer && segment && segment->format == GST_FORMAT_TIME && gst_video_info_from_caps(&info, caps) &&
            GST_VIDEO_INFO_FORMAT(&info) == GST_VIDEO_FORMAT_NV12 &&
            ((GST_VIDEO_INFO_WIDTH(&info) == 640 && GST_VIDEO_INFO_HEIGHT(&info) == 360) ||
             (GST_VIDEO_INFO_WIDTH(&info) == 3840 && GST_VIDEO_INFO_HEIGHT(&info) == 2160)) &&
            GST_BUFFER_PTS_IS_VALID(buffer)) {
            auto stamp = gst_segment_to_running_time(segment, GST_FORMAT_TIME, GST_BUFFER_PTS(buffer));
            GstVideoFrame frame{};
            if (stamp != GST_CLOCK_TIME_NONE && stamp <= static_cast<GstClockTime>(std::numeric_limits<std::int64_t>::max()) &&
                gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
                const auto stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
                if (stride >= GST_VIDEO_INFO_WIDTH(&info)) {
                    try {
                        cv::Mat gray(GST_VIDEO_INFO_HEIGHT(&info), GST_VIDEO_INFO_WIDTH(&info), CV_8UC1,
                                     GST_VIDEO_FRAME_PLANE_DATA(&frame, 0), static_cast<std::size_t>(stride));
                        accepted = queue_.push(token, camera, gray, static_cast<std::int64_t>(stamp), now_ns());
                        if (!accepted) fault = true;
                    }
                    catch (...) { fault = true; }
                } else fault = true;
                gst_video_frame_unmap(&frame);
            } else fault = true;
        } else fault = true;
        gst_sample_unref(sample);
    }
    if (!sample && gst_app_sink_is_eos(sink)) fault = true;
    if (fault) { stop_if_current(token); accepted = false; }
    else if (!check_bus(token, pipeline)) accepted = false;
    gst_object_unref(sink); gst_object_unref(pipeline);
    return accepted;
}
std::optional<StereoPair> GstStereoCapture::select(bool require_sync) {
    GstElement* pipeline;
    WorkToken token;
    {
        std::lock_guard<std::mutex> lock(control_);
        if (!pipeline_ || !token_) return {};
        token = *token_;
        pipeline = GST_ELEMENT(gst_object_ref(pipeline_));
    }
    const bool healthy = check_bus(token, pipeline);
    gst_object_unref(pipeline);
    return healthy ? queue_.select(now_ns(), require_sync) : std::nullopt;
}
bool GstStereoCapture::is_running() const {
    std::lock_guard<std::mutex> lock(control_);
    return pipeline_ != nullptr;
}
} // namespace agtc
