#pragma once
#include "preview_types.h"
#include <httplib.h>
#include <atomic>
#include <mutex>
#include <thread>

namespace agtc {
// Candidate loopback HTTP endpoint; single-use, stop before store destruction.
// This does not implement the prior dashboard, calibration or video routes.
class DepthHttpServer {
public:
    explicit DepthHttpServer(DepthStore& store, std::function<void()> on_valid_request = {},
                             std::function<PreviewResult()> preview = {},
                             std::function<StereoPreviewResult()> stereo_preview = {});
    ~DepthHttpServer();
    DepthHttpServer(const DepthHttpServer&) = delete;
    DepthHttpServer& operator=(const DepthHttpServer&) = delete;
    // port=0 requests an ephemeral port; returns bound port or -1 on failure.
    int start(int port = 0);
    void stop();
private:
    DepthStore& store_;
    std::function<void()> on_valid_request_;
    std::function<PreviewResult()> preview_;
    std::function<StereoPreviewResult()> stereo_preview_;
    httplib::Server server_;
    std::thread thread_;
    std::mutex lifecycle_;
    bool started_ = false;
    std::atomic<bool> socket_options_ok_{true};
};
} // namespace agtc
