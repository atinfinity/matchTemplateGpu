#pragma once

#include <opencv2/core.hpp>

double launchMatchTemplateCpu(const cv::Mat& img, const cv::Mat& templ, cv::Mat& result, const int loop_num);
double launchMatchTemplateCV(const cv::Mat& img, const cv::Mat& templ, cv::Mat& result, const int loop_num);
