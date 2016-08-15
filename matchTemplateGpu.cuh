#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

void launchMatchTemplateGpu(cv::cuda::GpuMat& img, cv::cuda::GpuMat& templ, cv::cuda::GpuMat& result);
