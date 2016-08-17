#include "matchTemplateGpu.cuh"

#include <opencv2/cudev.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void matchTemplateGpu
(
    const cv::cudev::PtrStepSz<uchar> img, 
    const cv::cudev::PtrStepSz<uchar> templ, 
    cv::cudev::PtrStepSz<float> result
)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < result.cols && y < result.rows){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (img.ptr((y + yy))[x + xx] - templ.ptr(yy)[xx]);
                sum += diff*diff;
            }
        }
        result.ptr(y)[x] = sum;
    }
}

void launchMatchTemplateGpu
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result
)
{
    cv::cudev::PtrStepSz<uchar> pImg =
        cv::cudev::PtrStepSz<uchar>(img.rows, img.cols * img.channels(), img.ptr<uchar>(), img.step);

    cv::cudev::PtrStepSz<uchar> pDst =
        cv::cudev::PtrStepSz<uchar>(templ.rows, templ.cols * templ.channels(), templ.ptr<uchar>(), templ.step);

    cv::cudev::PtrStepSz<float> pResult =
        cv::cudev::PtrStepSz<float>(result.rows, result.cols * result.channels(), result.ptr<float>(), result.step);

    const dim3 block(64, 2);
    const dim3 grid(cv::cudev::divUp(result.cols, block.x), cv::cudev::divUp(result.rows, block.y));

    matchTemplateGpu<<<grid, block>>>(pImg, pDst, pResult);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}
