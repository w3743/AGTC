#include "stereo_pair_queue.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    agtc::DepthStore store(2);
    store.set_geometry_valid(true);
    agtc::StereoPairQueue queue(store);
    auto generation = queue.start();
    assert(generation && !queue.start());
    cv::Mat padded(360, 648, CV_8UC1, cv::Scalar(42));
    cv::Mat left = padded(cv::Rect(4, 0, 640, 360));
    cv::Mat right(360, 640, CV_8UC1, cv::Scalar(77));
    assert(!left.isContinuous());
    assert(!queue.push(*generation, 2, left, 1, 1));
    assert(!queue.push(*generation, 0, left, -1, 1));
    assert(queue.push(*generation, 0, left, 100'000'000, 1'000'000'000));
    left.setTo(0);
    assert(queue.push(*generation, 1, right, 95'000'000, 1'000'000'001));
    assert(queue.push(*generation, 1, right, 105'000'000, 1'000'000'002));
    auto pair = queue.select(1'000'000'003);
    assert(pair && pair->left.gray.isContinuous() && pair->left.gray.at<std::uint8_t>(0, 0) == 42);
    assert(pair->right.pts_ns == 95'000'000); // Python's min() keeps first on equal skew.
    assert(!queue.push(*generation, 1, right, 104'000'000, 1'000'000'003));
    assert(!queue.push(*generation, 1, right, 106'000'000, 999'999'999));
    assert(!queue.select(999'999'999)); // future arrivals are not fresh
    assert(queue.select(1'750'000'000)); // 750 ms inclusive for left
    assert(!queue.select(1'750'000'001));
    assert(queue.push(*generation, 0, left, 200'000'000, 1'800'000'000));
    assert(queue.push(*generation, 1, right, 230'000'001, 1'800'000'000));
    assert(!queue.select(1'800'000'000)); // no candidate within 25 ms
    auto preview_pair = queue.select(1'800'000'000, false);
    assert(preview_pair && preview_pair->right.pts_ns - preview_pair->left.pts_ns == 30'000'001);
    queue.stop();
    assert(!queue.select(1'800'000'000));
    auto newer = queue.start();
    assert(newer && newer->epoch != generation->epoch);
    assert(!queue.push(*generation, 0, left, 1, 1));
    assert(queue.push(*newer, 0, left, 1, 1));
    assert(queue.push(*newer, 1, right, 1, 1));
    assert(queue.select(1));
    store.set_calibrating(true);
    assert(!queue.select(1));
    assert(!queue.push(*newer, 0, left, 2, 2));
    queue.stop();
}
