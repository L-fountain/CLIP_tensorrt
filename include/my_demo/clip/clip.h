#ifndef CLIP_H
#define CLIP_H

#include <filesystem>
#include <vector>
#include <unordered_set>

#include "common.h"
#include "parserOnnxConfig.h"
#include "argsParser.h"
#include "buffers.h"
#include "logger.h"
#include "NvInfer.h"

#include <cuda_runtime_api.h>
#include "clip_tokenizer.h"


using samplesCommon::SampleUniquePtr;

//! \brief  The CLIP class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class CLIP_Base
{
public:
    CLIP_Base(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    virtual bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    virtual bool infer();

protected:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an CLIP model for retrieval and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, 
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    virtual bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    virtual bool verifyOutput(const samplesCommon::BufferManager& buffers);
};



// For the specific implementation of vision
class CLIP_Vision: public CLIP_Base{
public:
    CLIP_Vision(const samplesCommon::VisionParams& params):CLIP_Base(params),
    image_mean(params.image_mean),image_std(params.image_std),USE_GPU_PREPROCESS(true){
        read_image(params.image_src);
    }

    virtual bool build() override;
    virtual bool infer() override;
    bool read_image(const std::filesystem::path&);

    virtual bool processInput(const samplesCommon::BufferManager& buffers) override;
    virtual bool verifyOutput(const samplesCommon::BufferManager& buffers) override;
    
private:
    std::vector<float> image_mean;
    std::vector<float> image_std;
    std::vector<std::string> image_paths;
    std::vector<std::vector<float>> image_features;
    bool USE_GPU_PREPROCESS;
    // 使用set存储支持的扩展名
    std::unordered_set<std::string> supportedExtensions = {".jpg", ".jpeg", ".png", ".bmp"};
};



// For the special implementation of language
class CLIP_Text: public CLIP_Base{
public:
    CLIP_Text(const samplesCommon::LanguageParams& params):CLIP_Base(params),
    tokenizer(params.vocab_path),text_token_length(params.text_token_length){
        read_text(params.text_src);
    }

    virtual bool build() override;
    virtual bool infer() override;
    bool read_text(const std::string&);

    virtual bool processInput(const samplesCommon::BufferManager& buffers) override;
    virtual bool verifyOutput(const samplesCommon::BufferManager& buffers) override;

private:
    CLIPTokenizer tokenizer;
    uint8_t text_token_length;
    std::vector<std::string> texts;
    std::vector<std::vector<float>> text_features;
};


// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;
    uint8_t * data;
    size_t size;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;
    float * data;
    size_t size;
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};



#endif