#pragma once
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace agtc {
struct StereoOutput { cv::Mat disparity; cv::Mat depth; cv::Mat valid; cv::Mat left_rectified; cv::Mat right_rectified; cv::Mat left_resized; cv::Mat right_resized; };

// Fixed 0.375 profile of the deployed Python FastSGBMDepth. Construction and
// compute may throw on invalid calibration/input or OpenCV errors. Not thread-safe.
class StereoSgbm {
public:
    explicit StereoSgbm(const std::string& calibration_yaml);
    StereoOutput compute(const cv::Mat& left_gray, const cv::Mat& right_gray);
    const std::string& calibration_sha256() const { return source_sha256_; }
    double depth_factor() const { return depth_factor_; }
    double disparity_offset() const { return disparity_offset_; }
    const std::array<double, 9>& wire_intrinsics() const { return wire_intrinsics_; }
    const cv::Mat& map_x(int camera) const { if (camera < 0 || camera > 1) throw std::out_of_range("camera"); return maps_x_[camera]; }
    const cv::Mat& map_y(int camera) const { if (camera < 0 || camera > 1) throw std::out_of_range("camera"); return maps_y_[camera]; }
private:
    cv::Mat rectify(const cv::Mat& source, int camera, cv::Mat& resized) const;
    cv::Mat maps_x_[2], maps_y_[2], support_[2];
    cv::Ptr<cv::StereoSGBM> matcher_;
    std::string source_sha256_;
    std::array<double, 9> wire_intrinsics_{};
    std::vector<std::uint8_t> filter_workspace_;
    double depth_factor_ = 0, disparity_offset_ = 0;
    int num_disparities_ = 0;
};
} // namespace agtc
