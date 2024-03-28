#include "preprocess.h"

// 封装的话, 可以放到Cpp之中; 结构体的定义可以放到H文件之中
Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type)
{
    Norm out;
    out.normType_    = NormType::MeanStd;
    out.alpha_       = alpha;
    out.channelType_ = channel_type;
    memcpy(out.mean_, mean, sizeof(out.mean_));
    memcpy(out.std_,  std,  sizeof(out.std_));
    return out;
}

Norm Norm::None()
{
    return Norm();
}

/**
 *@brief: out = (x * alpha - mean) / std, 标准化(减均值除方差)
* alpha 为 1; 
*@return: 3通道
*/ 
void StandNorm_c3(void* model_input_buffer, cv::Mat& src_mat, const Norm& in, const cv::Size& model_size)
{   
    cv::Mat img_cvtColr;    // 需要的话, 进行通道转换, bgr2rgb 或者 rgb2bgr
    cv::Mat img_convert;    // 利用cv::Mat的 convertTo 进行归一化除以255.0f
    cv::Mat norm_img;       // 存储减均值除方之后的数据
    cv::Mat img_resize;     // 是否需要进行 resize
    int nModelChannels = 3;

    if (in.channelType_ == ChannelType::Invert)
    {   
        cv::cvtColor(src_mat, img_cvtColr, cv::COLOR_BGR2RGB);  // inplace
    }

    if (src_mat.cols != model_size.width || src_mat.rows != model_size.height)
    {   
        //! 需要resize 的情况是 原图resize 还是 cvtColor的resize, 因此要判断是否需要 Invert
        if (in.channelType_ == ChannelType::Invert)
        {
            cv::resize(img_cvtColr, img_resize, model_size);
        }
        else
        {
            cv::resize(src_mat, img_resize, model_size);
        }

        img_resize.convertTo(img_convert, CV_32FC3, in.alpha_);
    }
    else
    {   
        //! 无需resize; 原图convert 还是 cvtColor的convert, 因此要判断是否需要 Invert
        if (in.channelType_ == ChannelType::Invert)
        {
            img_cvtColr.convertTo(img_convert, CV_32FC3, in.alpha_);
        }
        else
        {
            src_mat.convertTo(img_convert, CV_32FC3, in.alpha_);
        }
    }


    cv::Scalar mean(in.mean_[0], in.mean_[1], in.mean_[2]);
    cv::Scalar std(in.std_[0], in.std_[1], in.std_[2]);
    cv::Mat mean_mat(model_size, CV_32FC3, mean);
    cv::Mat std_mat(model_size, CV_32FC3, std);
    norm_img = (img_convert - mean_mat) / std_mat;

    //! 如果模型输入是fp16数据类型, 则需要在这里进行转换
    // cv:: fp16norm_img;
    // norm_img.convertTo(fp16norm_img, CV_16FC3);

    std::vector<cv::Mat> imgArray(nModelChannels);
    //! hwc2chw
    cv::split(norm_img, imgArray);

    //! 这里我们采用的是 flaot32 数据类型
    size_t dst_plane_size = model_size.area() * sizeof(float);
    memcpy(reinterpret_cast<uint8_t*>(model_input_buffer), 
        imgArray[0].data, imgArray[0].step.p[0]*imgArray[0].size[0]);
    memcpy(reinterpret_cast<uint8_t*>(model_input_buffer) + dst_plane_size, 
        imgArray[1].data, imgArray[1].step.p[0]*imgArray[1].size[0]);
    memcpy(reinterpret_cast<uint8_t*>(model_input_buffer) + 2*dst_plane_size, 
        imgArray[2].data, imgArray[2].step.p[0]*imgArray[2].size[0]);
    return;
}

