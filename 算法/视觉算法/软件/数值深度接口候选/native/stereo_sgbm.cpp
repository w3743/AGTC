#include "stereo_sgbm.h"
#include "depth_filter.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace agtc {
namespace {
cv::Mat matrix(const cv::FileStorage& fs, const char* name, int rows, int cols) {
    cv::Mat m;
    fs[name] >> m;
    if (m.empty() || m.type() != CV_64FC1 || m.rows != rows || m.cols != cols || !cv::checkRange(m))
        throw std::invalid_argument(std::string("invalid calibration matrix ") + name);
    return m;
}
} // namespace

StereoSgbm::StereoSgbm(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) throw std::invalid_argument("cannot open calibration");
    std::string schema, status, unit;
    int width = 0, height = 0;
    fs["schema"] >> schema; fs["status"] >> status; fs["unit"] >> unit;
    fs["source_sha256"] >> source_sha256_; fs["image_width"] >> width; fs["image_height"] >> height;
    if (schema != "agtc-stereo-calibration-v1" || status != "geometry_pass_metric_accuracy_unverified" ||
        unit != "mm" || width != 3840 || height != 2160 || source_sha256_.size() != 64 ||
        source_sha256_.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::invalid_argument("invalid calibration identity or profile");
    auto k1 = matrix(fs, "K1", 3, 3); auto k2 = matrix(fs, "K2", 3, 3);
    auto r1 = matrix(fs, "R1", 3, 3); auto r2 = matrix(fs, "R2", 3, 3);
    auto p1 = matrix(fs, "P1", 3, 4); auto p2 = matrix(fs, "P2", 3, 4);
    matrix(fs, "T", 3, 1);
    cv::Mat d1, d2; fs["D1"] >> d1; fs["D2"] >> d2;
    auto distortion_ok = [](const cv::Mat& d) {
        auto n = d.total();
        return !d.empty() && d.type() == CV_64FC1 && (d.rows == 1 || d.cols == 1) &&
               (n == 4 || n == 5 || n == 8 || n == 12 || n == 14) && cv::checkRange(d);
    };
    if (!distortion_ok(d1) || !distortion_ok(d2) || std::abs(p2.at<double>(1, 3)) <= std::abs(p2.at<double>(0, 3)) || p2.at<double>(1, 3) >= 0)
        throw std::invalid_argument("invalid stereo calibration or baseline");
    constexpr double scale = 240.0 / 3840.0;
    cv::Mat resize = cv::Mat(cv::Matx33d(scale, 0, (scale - 1) / 2, 0, scale, (scale - 1) / 2, 0, 0, 1));
    cv::Mat ks[] = {k1, k2}, ds[] = {d1, d2}, rs[] = {r1, r2}, ps[] = {p1, p2};
    for (int i = 0; i < 2; ++i) {
        cv::Mat scaled_k = resize * ks[i], scaled_p = resize * ps[i];
        cv::initUndistortRectifyMap(scaled_k, ds[i], rs[i], scaled_p, cv::Size(240, 135), CV_32FC1, maps_x_[i], maps_y_[i]);
        cv::Mat mask(135, 240, CV_8UC1, cv::Scalar(255)), remapped, upright, binary;
        cv::remap(mask, remapped, maps_x_[i], maps_y_[i], cv::INTER_LINEAR);
        cv::rotate(remapped, upright, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::compare(upright(cv::Rect(2, 0, 131, 240)), 254, binary, cv::CMP_GT);
        cv::erode(binary, support_[i], cv::Mat::ones(5, 5, CV_8UC1), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    cv::Mat sp1 = resize * p1, sp2 = resize * p2;
    wire_intrinsics_ = {sp1.at<double>(1, 1), 0.0, sp1.at<double>(1, 2) - 2,
                        0.0, sp1.at<double>(0, 0), 239.0 - sp1.at<double>(0, 2),
                        0.0, 0.0, 1.0};
    if (!std::isfinite(wire_intrinsics_[0]) || wire_intrinsics_[0] <= 0 ||
        !std::isfinite(wire_intrinsics_[2]) || !std::isfinite(wire_intrinsics_[4]) ||
        wire_intrinsics_[4] <= 0 || !std::isfinite(wire_intrinsics_[5]))
        throw std::invalid_argument("invalid derived depth intrinsics");
    depth_factor_ = -sp2.at<double>(1, 3) / 1000.0;
    disparity_offset_ = sp1.at<double>(1, 2) - sp2.at<double>(1, 2);
    const double required = disparity_offset_ + depth_factor_ / .3 + 4;
    if (!std::isfinite(depth_factor_) || depth_factor_ <= 0 || !std::isfinite(disparity_offset_) || !std::isfinite(required))
        throw std::invalid_argument("invalid depth factor or disparity offset");
    num_disparities_ = required >= 112 ? 112 : required <= 16 ? 16 : static_cast<int>(std::ceil(required / 16.0)) * 16;
    matcher_ = cv::StereoSGBM::create(0, num_disparities_, 5, 200, 800, 1, 31, 8, 40, 2, cv::StereoSGBM::MODE_SGBM_3WAY);
    const auto scratch_size = agtc_metric_depth_workspace_size(131, 240);
    if (!scratch_size) throw std::runtime_error("invalid native filter workspace size");
    filter_workspace_.resize(scratch_size);
}

cv::Mat StereoSgbm::rectify(const cv::Mat& source, int camera, cv::Mat& small) const {
    if (source.type() != CV_8UC1 || (source.size() != cv::Size(3840, 2160) && source.size() != cv::Size(640, 360)))
        throw std::invalid_argument("expected calibrated sensor frame or 640x360 full-FOV gray frame");
    cv::Mat remapped, upright;
    cv::resize(source, small, cv::Size(240, 135), 0, 0, cv::INTER_AREA);
    cv::remap(small, remapped, maps_x_[camera], maps_y_[camera], cv::INTER_LINEAR);
    cv::rotate(remapped, upright, cv::ROTATE_90_COUNTERCLOCKWISE);
    return upright(cv::Rect(2, 0, 131, 240)).clone();
}

StereoOutput StereoSgbm::compute(const cv::Mat& left, const cv::Mat& right) {
    cv::Mat left_resized, right_resized;
    cv::Mat l = rectify(left, 0, left_resized), r = rectify(right, 1, right_resized), lp, rp, raw, disparity;
    cv::copyMakeBorder(l, lp, 0, 0, num_disparities_, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::copyMakeBorder(r, rp, 0, 0, num_disparities_, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
    matcher_->compute(lp, rp, raw);
    raw(cv::Rect(num_disparities_, 0, 131, 240)).convertTo(disparity, CV_32F, 1.0 / 16.0);
    cv::Mat depth(240, 131, CV_32F), valid(240, 131, CV_8U);
    if (agtc_metric_depth_f32_with_workspace(disparity.ptr<float>(), support_[0].ptr<std::uint8_t>(), support_[1].ptr<std::uint8_t>(),
                              131, 240, disparity_offset_, depth_factor_, depth.ptr<float>(), valid.ptr<std::uint8_t>(),
                              filter_workspace_.data(), filter_workspace_.size()) != 0)
        throw std::runtime_error("native depth filter failed");
    return {std::move(disparity), std::move(depth), std::move(valid), std::move(l), std::move(r), std::move(left_resized), std::move(right_resized)};
}
} // namespace agtc
