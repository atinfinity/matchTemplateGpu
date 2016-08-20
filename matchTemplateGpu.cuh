#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

double launchMatchTemplateGpu(cv::cuda::GpuMat& img, cv::cuda::GpuMat& templ, cv::cuda::GpuMat& result, const int loop_num);

// use shared memory
double launchMatchTemplateGpu_opt(cv::cuda::GpuMat& img, cv::cuda::GpuMat& templ, cv::cuda::GpuMat& result, const int loop_num);

// use shared memory + __ldg
double launchMatchTemplateGpu_opt2(cv::cuda::GpuMat& img, cv::cuda::GpuMat& templ, cv::cuda::GpuMat& result, const int loop_num);
