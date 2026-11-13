#include "depth_http_server.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <future>

using namespace std::chrono_literals;
int main() {
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    store.set_capture_running(true);
    std::atomic<int> preview_calls{0};
    agtc::DepthHttpServer server(store, {}, [&] {
        ++preview_calls;
        auto source = std::make_shared<const agtc::DepthFrame>(agtc::DepthFrame{1, 1, 1, {}, "{}\r\nInjected: yes"});
        auto frame = std::make_shared<const agtc::PreviewFrame>(agtc::PreviewFrame{source, {0xff, 0xd8, 0xff, 0xd9}});
        return agtc::PreviewResult{agtc::DepthStatus::ok, frame};
    });
    const int port = server.start();
    assert(port > 0);
    agtc::DepthHttpServer duplicate(store);
    assert(duplicate.start(port) == -1);
    httplib::Client client("127.0.0.1", port);
    auto invalid = client.Get("/depth.bin?sequence=bad");
    assert(invalid && invalid->status == 400);
    auto range = client.Get("/depth.bin", {{"Range", "bytes=0-3"}});
    assert(range && range->status == 400);
    auto unsupported = client.Get("/depth.jpg?scale=auto");
    assert(unsupported && unsupported->status == 400 && preview_calls == 0);
    auto injected = client.Get("/depth.jpg?scale=fixed");
    assert(injected && injected->status == 503 && !injected->has_header("Injected") && preview_calls == 1);
    auto pending = std::async(std::launch::async, [port] {
        httplib::Client other("127.0.0.1", port);
        auto result = other.Get("/depth.bin");
        return result ? result->status : -1;
    });
    assert(pending.wait_for(50ms) == std::future_status::timeout);
    const auto started = std::chrono::steady_clock::now();
    server.stop();
    assert(std::chrono::steady_clock::now() - started < 5s);
    assert(pending.get() == 503);
    assert(server.start() == -1); // single-use lifecycle
}
