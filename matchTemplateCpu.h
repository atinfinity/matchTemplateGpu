#pragma once

#include <opencv2/core.hpp>

void matchTemplateCpu(const cv::Mat& img, const cv::Mat& templ, cv::Mat& result);
