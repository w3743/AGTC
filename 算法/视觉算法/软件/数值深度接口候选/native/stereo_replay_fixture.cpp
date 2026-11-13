// Offline raw-frame fixture; not a vehicle executable or a network service.
#include "stereo_sgbm.h"
#include "stereo_publisher.h"
#include "stereo_pair_queue.h"
#include "depth_http.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

static cv::Mat read_gray(const char* path, int width, int height) {
    cv::Mat image(height, width, CV_8UC1);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(image.data), static_cast<std::streamsize>(image.total()));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("invalid raw input length");
    return image;
}
static void save(const std::string& path, const cv::Mat& image) {
    if (!image.isContinuous()) throw std::runtime_error("non-contiguous output");
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(image.data), static_cast<std::streamsize>(image.total() * image.elemSize()));
    if (!output) throw std::runtime_error("output write failed");
}
int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) { std::cerr << "usage: fixture calibration.yml left.raw right.raw width height output-prefix [runs]\n"; return 2; }
    try {
        int width = std::stoi(argv[4]), height = std::stoi(argv[5]);
        if (!((width == 640 && height == 360) || (width == 3840 && height == 2160))) throw std::invalid_argument("invalid input dimensions");
        const int runs = argc == 8 ? std::stoi(argv[7]) : 1;
        if (runs < 1 || runs > 10000) throw std::invalid_argument("invalid run count");
        cv::setNumThreads(2);
        agtc::StereoSgbm engine(argv[1]);
        auto left = read_gray(argv[2], width, height), right = read_gray(argv[3], width, height);
        for (int i = 0; i < (runs > 1 ? 5 : 0); ++i) engine.compute(left, right);
        std::vector<std::int64_t> timings;
        agtc::StereoOutput result;
        for (int i = 0; i < runs; ++i) {
            const auto start = std::chrono::steady_clock::now();
            result = engine.compute(left, right);
            timings.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        }
        const std::string prefix(argv[6]);
        save(prefix + ".disparity", result.disparity);
        save(prefix + ".depth", result.depth);
        save(prefix + ".valid", result.valid);
        save(prefix + ".left_rectified", result.left_rectified);
        save(prefix + ".right_rectified", result.right_rectified);
        save(prefix + ".left_resized", result.left_resized);
        save(prefix + ".right_resized", result.right_resized);
        save(prefix + ".left_map_x", engine.map_x(0));
        save(prefix + ".left_map_y", engine.map_y(0));
        std::ofstream timing_file(prefix + ".timings_ns");
        for (auto ns : timings) timing_file << ns << '\n';
        if (!timing_file) throw std::runtime_error("timing write failed");
        auto clock = [] { return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count(); };
        agtc::DepthStore store(2);
        store.set_geometry_valid(true);
        agtc::StereoPairQueue queue(store);
        auto token_result = queue.start();
        if (!token_result) throw std::runtime_error("capture queue start failed");
        auto token = *token_result;
        agtc::StereoPublisher publisher(engine, store, "replay-fixture", "fixture-steady-clock", clock);
        auto arrival = clock();
        agtc::TimedStereoFrame l{left, 1'000'000'000, arrival}, r{right, 1'010'000'000, arrival};
        if (!queue.push(token, 0, l.gray, l.pts_ns, l.arrival_ns) ||
            !queue.push(token, 1, r.gray, r.pts_ns, r.arrival_ns)) throw std::runtime_error("capture queue push failed");
        auto pair = queue.select(clock());
        if (!pair || publisher.process(pair->token, pair->left, pair->right) != agtc::PublishStatus::ok)
            throw std::runtime_error("stereo pair publish failed");
        auto latest = store.read({}, std::chrono::milliseconds(0), std::chrono::seconds(1));
        if (latest.status != agtc::DepthStatus::ok || !latest.frame) throw std::runtime_error("stereo read failed");
        std::ofstream wire(prefix + ".published", std::ios::binary);
        wire.write(reinterpret_cast<const char*>(latest.frame->payload.data()), static_cast<std::streamsize>(latest.frame->payload.size()));
        if (!wire) throw std::runtime_error("published depth write failed");
        std::ofstream meta(prefix + ".published.json");
        meta << latest.frame->metadata_ascii;
        if (!meta) throw std::runtime_error("metadata write failed");
        auto http = agtc::route_depth_get(store, "/depth.bin", std::chrono::milliseconds(0), std::chrono::seconds(1));
        if (http.status != 200 || !http.frame) throw std::runtime_error("published HTTP routing failed");
        std::ofstream response_file(prefix + ".http", std::ios::binary);
        response_file.write(http.headers.data(), static_cast<std::streamsize>(http.headers.size()));
        response_file.write(reinterpret_cast<const char*>(http.frame->payload.data()), static_cast<std::streamsize>(http.frame->payload.size()));
        if (!response_file) throw std::runtime_error("HTTP response write failed");
        if (publisher.process(token, l, r) != agtc::PublishStatus::duplicate)
            throw std::runtime_error("duplicate frame was republished");
        auto old = l; old.arrival_ns = clock() - 751'000'000;
        old.pts_ns += 1;
        if (publisher.process(token, old, r) != agtc::PublishStatus::stale) throw std::runtime_error("stale frame was not rejected");
        if (store.read({}, std::chrono::milliseconds(0), std::chrono::seconds(1)).status == agtc::DepthStatus::ok)
            throw std::runtime_error("stale frame left old data available");
        auto skewed = r; skewed.pts_ns = l.pts_ns + 25'000'001;
        if (publisher.process(token, l, skewed) != agtc::PublishStatus::skewed) throw std::runtime_error("skewed pair was not rejected");
        auto future = l; future.arrival_ns = clock() + 1'000'000'000;
        future.pts_ns += 1;
        if (publisher.process(token, future, r) != agtc::PublishStatus::stale) throw std::runtime_error("future frame was not rejected");
        arrival = clock(); l.arrival_ns = arrival; r.arrival_ns = arrival; l.pts_ns += 33'000'000; r.pts_ns += 33'000'000;
        if (publisher.process(token, l, r) != agtc::PublishStatus::ok) throw std::runtime_error("recovery publish failed");
        auto recovered = store.read({}, std::chrono::milliseconds(0), std::chrono::seconds(1));
        if (recovered.status != agtc::DepthStatus::ok || recovered.frame->sequence != 2)
            throw std::runtime_error("recovery sequence was incorrect");
        auto regressed = l; regressed.pts_ns -= 1;
        if (publisher.process(token, regressed, r) != agtc::PublishStatus::regressed)
            throw std::runtime_error("regressed timestamp was not rejected");
        if (store.read({}, std::chrono::milliseconds(0), std::chrono::seconds(1)).status == agtc::DepthStatus::ok)
            throw std::runtime_error("regressed timestamp left old data available");
        int late_calls = 0;
        agtc::StereoPublisher late(engine, store, "replay-fixture", "fixture-steady-clock", [&] {
            return ++late_calls == 1 ? arrival : arrival + 1'500'000'001;
        });
        if (late.process(token, l, r) != agtc::PublishStatus::stale) throw std::runtime_error("late result was not rejected");
        if (store.read({}, std::chrono::milliseconds(0), std::chrono::seconds(1)).status == agtc::DepthStatus::ok)
            throw std::runtime_error("late result left old data available");
        int calls = 0;
        agtc::StereoPublisher transitioning(engine, store, "replay-fixture", "fixture-steady-clock", [&] {
            if (++calls == 2) queue.stop();
            return clock();
        });
        arrival = clock(); l.arrival_ns = arrival; r.arrival_ns = arrival;
        if (transitioning.process(token, l, r) != agtc::PublishStatus::changed) throw std::runtime_error("pipeline transition was not rejected");
        std::cout << engine.calibration_sha256() << "\n" << CV_VERSION << "\n";
        return 0;
    } catch (const std::exception& exc) { std::cerr << exc.what() << "\n"; return 1; }
}
