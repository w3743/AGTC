#include "depth_http_server.h"
#include "depth_http.h"
#include <future>

namespace agtc {
DepthHttpServer::DepthHttpServer(DepthStore& store, std::function<void()> on_valid_request,
                                 std::function<PreviewResult()> preview,
                                 std::function<StereoPreviewResult()> stereo_preview)
    : store_(store), on_valid_request_(std::move(on_valid_request)), preview_(std::move(preview)),
      stereo_preview_(std::move(stereo_preview)) {
    server_.new_task_queue = [] { return new httplib::ThreadPool(4, 16); };
    // cpp-httplib's defaults may allow multiple listeners on the same port.
    server_.set_socket_options([this](socket_t socket) {
        const int enabled = 1;
#ifdef _WIN32
        if (setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                       reinterpret_cast<const char*>(&enabled), sizeof(enabled)) != 0) socket_options_ok_ = false;
#else
        if (setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled)) != 0) socket_options_ok_ = false;
#endif
    });
    server_.set_read_timeout(2, 0).set_write_timeout(2, 0).set_keep_alive_max_count(1).set_payload_max_length(0);
    server_.Get(R"(/depth\.bin)", [this](const httplib::Request& req, httplib::Response& res) {
        if (req.has_header("Range")) {
            res.status = 400;
            res.set_header("Cache-Control", "no-store");
            res.set_content("{\"ok\":false,\"state\":\"range_unsupported\"}", "application/json; charset=utf-8");
            return;
        }
        auto response = route_depth_get(store_, req.target, std::chrono::seconds(1),
                                        std::chrono::milliseconds(1500), on_valid_request_);
        res.status = response.status;
        res.set_header("Cache-Control", "no-store");
        if (!response.frame) {
            res.set_content(response.error_body, "application/json; charset=utf-8");
            return;
        }
        auto frame = response.frame;
        res.set_header("X-Depth-Metadata", frame->metadata_ascii);
        res.set_content_provider(frame->payload.size(), "application/octet-stream",
            [frame](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
                if (offset > frame->payload.size() || length > frame->payload.size() - offset) return false;
                return sink.write(reinterpret_cast<const char*>(frame->payload.data() + offset), length);
            });
    });
    server_.Get(R"(/depth\.jpg)", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
        if (req.has_header("Range") || req.get_param_value_count("scale") != 1 ||
            req.get_param_value("scale") != "fixed") {
            res.status = 400;
            res.set_content("{\"ok\":false,\"state\":\"unsupported_scale_or_range\"}", "application/json; charset=utf-8");
            return;
        }
        if (!preview_) {
            res.status = 503;
            res.set_content("{\"ok\":false,\"state\":\"preview_unavailable\"}", "application/json; charset=utf-8");
            return;
        }
        if (on_valid_request_) on_valid_request_();
        auto result = preview_();
        bool safe = result.frame && result.frame->source && !result.frame->jpeg.empty() &&
                    !result.frame->source->metadata_ascii.empty() && result.frame->source->metadata_ascii.size() <= 8192;
        if (safe) for (unsigned char c : result.frame->source->metadata_ascii)
            if (c < 0x20 || c > 0x7e) { safe = false; break; }
        if (result.status != DepthStatus::ok || !safe) {
            res.status = result.status == DepthStatus::calibrating ? 409 : 503;
            res.set_content("{\"ok\":false,\"state\":\"preview_unavailable\"}", "application/json; charset=utf-8");
            return;
        }
        auto frame = result.frame;
        res.status = 200;
        res.set_header("X-Depth-Metadata", frame->source->metadata_ascii);
        res.set_header("X-Depth-Sequence", std::to_string(frame->source->sequence));
        res.set_content_provider(frame->jpeg.size(), "image/jpeg",
            [frame](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
                if (offset > frame->jpeg.size() || length > frame->jpeg.size() - offset) return false;
                return sink.write(reinterpret_cast<const char*>(frame->jpeg.data() + offset), length);
            });
    });
    server_.Get(R"(/stereo\.jpg)", [this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
        if (req.has_header("Range")) {
            res.status = 400;
            res.set_content("{\"ok\":false,\"state\":\"range_unsupported\"}", "application/json; charset=utf-8");
            return;
        }
        auto result = stereo_preview_ ? stereo_preview_() : StereoPreviewResult{};
        if (!result.frame || result.frame->jpeg.empty() || result.frame->left_pts_ns < 0 || result.frame->right_pts_ns < 0) {
            res.status = 503;
            res.set_content("{\"ok\":false,\"state\":\"preview_unavailable\"}", "application/json; charset=utf-8");
            return;
        }
        auto frame = result.frame;
        const auto skew = frame->left_pts_ns >= frame->right_pts_ns ?
            frame->left_pts_ns - frame->right_pts_ns : frame->right_pts_ns - frame->left_pts_ns;
        res.status = 200;
        res.set_header("X-Stereo-Left-Pts-Ns", std::to_string(frame->left_pts_ns));
        res.set_header("X-Stereo-Right-Pts-Ns", std::to_string(frame->right_pts_ns));
        res.set_header("X-Stereo-Pair-Skew-Ns", std::to_string(skew));
        res.set_header("X-Stereo-Exposure-Sync", "unverified");
        res.set_content_provider(frame->jpeg.size(), "image/jpeg",
            [frame](std::size_t offset, std::size_t length, httplib::DataSink& sink) {
                if (offset > frame->jpeg.size() || length > frame->jpeg.size() - offset) return false;
                return sink.write(reinterpret_cast<const char*>(frame->jpeg.data() + offset), length);
            });
    });
}
DepthHttpServer::~DepthHttpServer() { stop(); }
int DepthHttpServer::start(int port) {
    std::lock_guard<std::mutex> lock(lifecycle_);
    if (started_ || port < 0 || port > 65535) return -1;
    started_ = true;
    auto promise = std::make_shared<std::promise<int>>();
    auto ready = promise->get_future();
    thread_ = std::thread([this, port, promise] {
        int actual = -1;
        try {
            actual = port == 0 ? server_.bind_to_any_port("127.0.0.1") :
                                  (server_.bind_to_port("127.0.0.1", port) ? port : -1);
        } catch (...) { actual = -1; }
        promise->set_value(actual > 0 && !socket_options_ok_ ? -2 : actual);
        if (actual > 0) {
            try { server_.listen_after_bind(); } catch (...) {}
        }
        server_.decommission();
    });
    const int actual = ready.get();
    if (actual == -2) {
        server_.wait_until_ready();
        if (server_.is_running()) server_.stop();
        if (thread_.joinable()) thread_.join();
        return -1;
    }
    if (actual <= 0) { if (thread_.joinable()) thread_.join(); return -1; }
    server_.wait_until_ready();
    if (!server_.is_running()) { if (thread_.joinable()) thread_.join(); return -1; }
    return actual;
}
void DepthHttpServer::stop() {
    std::lock_guard<std::mutex> lock(lifecycle_);
    if (!thread_.joinable()) return;
    server_.stop();
    thread_.join();
}
} // namespace agtc
