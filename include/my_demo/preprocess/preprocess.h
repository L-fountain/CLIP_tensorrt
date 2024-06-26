#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>

/*
srcImg:    source image for inference
dstData:   data after preprocess (resize / bgr to rgb / hwc to chw / normalize)
dstHeight: model input height
dstWidth:  model input width
*/
void preprocess(const cv::Mat& srcImg, const int dstHeight, const int dstWidth, float* dstDevData);

void init_constants(std::vector<float>&, std::vector<float>&);

enum class NormType : int{
    None      = 0,
    MeanStd   = 1,
    AlphaBeta = 2
};

enum class ChannelType : int{
    None          = 0,
    Invert        = 1
};

/**
 * \brief: 为了预处理方便, 我们对预处理需要参数进行封装;
 * 
 */ 
struct Norm{
    float mean_[3];
    float std_[3];
    float alpha_;
    float beta_;

    NormType    normType_    = NormType::None;
    ChannelType channelType_ = ChannelType::None;

    /**
     * \brief: 静态方法, 可以不实例化 Norm, 仅仅通过 Norm::mean_std() 来调用; 减均值除方差的参数赋值
     */ 
    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);

    // None
    static Norm None();
};


void StandNorm_c3(void* model_input_buffer, cv::Mat& src_mat,const Norm& in,const cv::Size& model_size);

#endif  // PREPROCESS_H