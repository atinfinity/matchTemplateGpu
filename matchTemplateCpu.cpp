#include "matchTemplateCpu.h"
#include <opencv2/imgproc.hpp>

void matchTemplateCpu
(
    const cv::Mat& img, 
    const cv::Mat& templ, 
    cv::Mat& result
)
{
    for(int y = 0; y < result.rows; y++){
        float* presult = result.ptr<float>(y);
        for(int x = 0; x < result.cols; x++){
            long sum = 0;
            for(int yy = 0; yy < templ.rows; yy++){
                const uchar* pimg   = img.ptr<uchar>(y + yy);
                const uchar* ptempl = templ.ptr<uchar>(yy);
                for(int xx = 0; xx < templ.cols; xx++){
                    int diff = pimg[x + xx] - ptempl[xx];
                    sum += (diff*diff);
                }
            }
            presult[x] = sum;
        }
    }
}

double launchMatchTemplateCpu
(
    const cv::Mat& img, 
    const cv::Mat& templ, 
    cv::Mat& result, 
    const int loop_num
)
{
    double f = 1000.0f/cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for(int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        matchTemplateCpu(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

double launchMatchTemplateCV
(
    const cv::Mat& img,
    const cv::Mat& templ,
    cv::Mat& result,
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        cv::matchTemplate(img, templ, result, cv::TM_SQDIFF);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}
