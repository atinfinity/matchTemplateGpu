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
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if((x < result.cols) && (y < result.rows)){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (img.ptr((y+yy))[x+xx] - templ.ptr(yy)[xx]);
                sum += (diff*diff);
            }
        }
        result.ptr(y)[x] = sum;
    }
}

// use shared memory
__global__ void matchTemplateGpu_opt
(
    const cv::cudev::PtrStepSz<uchar> img,
    const cv::cudev::PtrStepSz<uchar> templ,
    cv::cudev::PtrStepSz<float> result
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ uchar temp[];

    if(threadIdx.x == 0){
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                temp[yy*templ.cols+xx] = templ.ptr(yy)[xx];
            }
        }
    }
    __syncthreads();

    if((x < result.cols) && (y < result.rows)){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (img.ptr((y+yy))[x+xx] - temp[yy*templ.cols+xx]);
                sum += (diff*diff);
            }
        }
        result.ptr(y)[x] = sum;
    }
}

// use shared memory
__global__ void matchTemplateGpu_opt2
(
    const cv::cudev::PtrStepSz<uchar> img,
    const cv::cudev::PtrStepSz<uchar> templ,
    cv::cudev::PtrStepSz<float> result
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    extern __shared__ uchar temp[];

    if(threadIdx.x == 0){
        for(int yy = 0; yy < templ.rows; yy++){
            const uchar* ptempl = templ.ptr(yy);
            for(int xx = 0; xx < templ.cols; xx++){
                temp[yy*templ.cols+xx]  = __ldg(&ptempl[xx]);
            }
        }
    }
    __syncthreads();

    if((x < result.cols) && (y < result.rows)){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            const uchar* pimg = img.ptr((y+yy)) + x;
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (__ldg(&pimg[xx]) - temp[yy*templ.cols+xx]);
                sum += (diff*diff);
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

// use shared memory
void launchMatchTemplateGpu_opt
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
    const size_t shared_mem_size = templ.cols*templ.rows*sizeof(uchar);

    matchTemplateGpu_opt<<<grid, block, shared_mem_size>>>(pImg, pDst, pResult);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

// use shared memory
void launchMatchTemplateGpu_opt2
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
    const size_t shared_mem_size = templ.cols*templ.rows*sizeof(uchar);

    matchTemplateGpu_opt2<<<grid, block, shared_mem_size>>>(pImg, pDst, pResult);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
    CV_CUDEV_SAFE_CALL(cudaDeviceSynchronize());
}

double launchMatchTemplateGpu
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

// use shared memory
double launchMatchTemplateGpu_opt
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu_opt(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

// use shared memory + __ldg
double launchMatchTemplateGpu_opt2
(
    cv::cuda::GpuMat& img, 
    cv::cuda::GpuMat& templ, 
    cv::cuda::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu_opt2(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}
