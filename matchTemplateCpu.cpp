#include "matchTemplateCpu.h"

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
