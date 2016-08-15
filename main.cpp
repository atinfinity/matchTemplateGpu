#include "matchTemplateCpu.h"
#include "matchTemplateGpu.cuh"
#include "utility.h"

#include <opencv2/imgproc.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    cv::Mat img(sz1080p, CV_8UC1, cv::Scalar(0));
    cv::Mat templ(cv::Size(32, 32), CV_8UC1, cv::Scalar(255));
    cv::Mat result, result_cv;

    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;

    // Naive Implementation
    start = cv::getTickCount();
    matchTemplateCpu(img, templ, result);
    end = cv::getTickCount();
    std::cout << "Naive: " << ((end - start) * f) << " ms." << std::endl;

    // OpenCV
    start = cv::getTickCount();
    cv::matchTemplate(img, templ, result_cv, cv::TM_SQDIFF);
    end = cv::getTickCount();
    std::cout << "OpenCV: " << ((end - start) * f) << " ms." << std::endl;

    cv::cuda::GpuMat d_img(img);
    cv::cuda::GpuMat d_templ(templ);
    cv::Size corrSize(img.cols - templ.cols + 1, img.rows - templ.rows + 1);
    cv::cuda::GpuMat d_result(corrSize, CV_32FC1, cv::Scalar(0.0f));

    // CUDA Implementation
    start = cv::getTickCount();
    launchMatchTemplateGpu(d_img, d_templ, d_result);
    end = cv::getTickCount();
    std::cout << "CUDA: " << ((end - start) * f) << " ms." << std::endl;
    std::cout << std::endl;

    // Verification
    verify(result, d_result);

    return 0;
}
