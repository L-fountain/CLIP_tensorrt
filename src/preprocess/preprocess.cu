#include "preprocess.h"


__constant__ float MEAN[3];
__constant__ float STD[3];

void init_constants(std::vector<float>& host_mean, std::vector<float>& host_std){
    // 将主机数据复制到设备常量内存
    cudaMemcpyToSymbol(MEAN, host_mean.data(), sizeof(MEAN));
    cudaMemcpyToSymbol(STD, host_std.data(), sizeof(STD));
}

// cuda image preprocess
__global__ void resize(const uchar* srcData, const int srcH, const int srcW, float* tgtData, const int tgtH, const int tgtW)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * tgtW;
    int idx3 = idx * 3;

    float scaleY = (float)tgtH / (float)srcH;
    float scaleX = (float)tgtW / (float)srcW;

    // (ix, iy)为目标图像坐标
    // (before_x, before_y)原图坐标
    float beforeX = float(ix + 0.5) / scaleX - 0.5;
    float beforeY = float(iy + 0.5) / scaleY - 0.5;
    // 原图像坐标四个相邻点
    // 获得变换前最近的四个顶点,取整
    int topY = static_cast<int>(beforeY);
    int bottomY = topY + 1;
    int leftX = static_cast<int>(beforeX);
    int rightX = leftX + 1;
    //计算变换前坐标的小数部分
    float u = beforeX - leftX;
    float v = beforeY - topY;
    uchar temp[3];
    if (ix < tgtW && iy < tgtH)
    {
        // resize
        // 如果计算的原始图像的像素大于真实原始图像尺寸
        if (topY >= srcH - 1 && leftX >= srcW - 1)  //右下角
        {
            for (int k = 0; k < 3; k++)
            {
                temp[idx3 + k] = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k];
            }
        }
        else if (topY >= srcH - 1)  // 最后一行
        {
            for (int k = 0; k < 3; k++)
            {
                temp[idx3 + k]
                = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
                + (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k];
            }
        }
        else if (leftX >= srcW - 1)  // 最后一列
        {
            for (int k = 0; k < 3; k++)
            {
                temp[idx3 + k]
                = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
                + (1. - u) * (v) * srcData[(leftX + bottomY * srcW) * 3 + k];
            }
        }
        else  // 非最后一行或最后一列情况
        {
            for (int k = 0; k < 3; k++)
            {
                temp[idx3 + k]
                = (1. - u) * (1. - v) * srcData[(leftX + topY * srcW) * 3 + k]
                + (u) * (1. - v) * srcData[(rightX + topY * srcW) * 3 + k]
                + (1. - u) * (v) * srcData[(leftX + bottomY * srcW) * 3 + k]
                + u * v * srcData[(rightX + bottomY * srcW) * 3 + k];
            }
        }

        // hwc to chw / bgr to rgb / normalize
        tgtData[idx] = ((float)temp[idx3 + 2] / 255.0 - MEAN[0]) / STD[0];
        tgtData[idx + tgtW * tgtH] = ((float)temp[idx3 + 1] / 255.0 - MEAN[1]) / STD[1];
        tgtData[idx + tgtW * tgtH * 2] = ((float)temp[idx3] / 255.0 - MEAN[2]) / STD[2];
    }
}


__global__ void process(const uchar* srcData, float* tgtData, const int h, const int w)
{
    /*
        (img / 255. - mean) / std
    */
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = ix + iy * w;
    int idx3 = idx * 3;

    // hwc to chw / bgr to rgb / normalize
    if (ix < w && iy < h)
    {
        tgtData[idx] = ((float)srcData[idx3 + 2] / 255.0 - 0.48145466) / 0.26862954;  // R pixel
        tgtData[idx + h * w] = ((float)srcData[idx3 + 1] / 255.0 - 0.4578275) / 0.26130258;  // G pixel
        tgtData[idx + h * w * 2] = ((float)srcData[idx3] / 255.0 - 0.40821073) / 0.27577711;  // B pixel
    }
}


/*
srcImg:    source image for inference
dstData:   data after preprocess (resize / bgr to rgb / hwc to chw / normalize)
dstHeight: model input height
dstWidth:  model input width
*/
void preprocess(const cv::Mat& srcImg, const int dstHeight, const int dstWidth, float* dstDevData)
{
    int srcHeight = srcImg.rows;
    int srcWidth = srcImg.cols;
    int srcElements = srcHeight * srcWidth * 3;
    int dstElements = dstHeight * dstWidth * 3;

    // target data on device 这里一步到位直接拿来开辟好的GPU内存了，所以不需要再开辟内存了
    // float* dstDevData;
    // cudaMalloc((void**)&dstDevData, sizeof(float) * dstElements);

    // 将两个核函数写在一起也就不需要开辟和拷贝中间内存了
    // middle image data on device ( for bilinear resize )
    // uchar* midDevData;
    // cudaMalloc((void**)&midDevData, sizeof(uchar) * dstElements);

    // source images data on device
    uchar* srcDevData;
    cudaMalloc((void**)&srcDevData, sizeof(uchar) * srcElements);
    cudaMemcpy(srcDevData, srcImg.ptr(), sizeof(uchar) * srcElements, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16); // 有待测试
    dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x, (dstHeight + blockSize.y - 1) / blockSize.y);

    // bilinear resize
    resize<<<gridSize, blockSize>>>(srcDevData, srcHeight, srcWidth, dstDevData, dstHeight, dstWidth);
    cudaDeviceSynchronize();

    //  hwc to chw / bgr to rgb / normalize 与上面的resize合并了
    // process<<<gridSize, blockSize>>>(midDevData, dstDevData, dstHeight, dstWidth);
    // cudaDeviceSynchronize();

    // 不需要将GPU内存拷贝回主机
    // cudaMemcpy(dstData, dstDevData, sizeof(float) * dstElements, cudaMemcpyDeviceToHost);

    cudaFree(srcDevData);
    // cudaFree(midDevData);
    // cudaFree(dstDevData);
}

