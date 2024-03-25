#ifndef CLIP_H
#define CLIP_H

#include "common.h"
#include "parserOnnxConfig.h"
#include "argsParser.h"
#include "buffers.h"
#include "logger.h"
#include "NvInfer.h"

#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"

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
    CLIP_Vision(const samplesCommon::OnnxSampleParams& params):CLIP_Base(params){}

    virtual bool build() override;
    virtual bool infer() override;
    virtual bool processInput(const samplesCommon::BufferManager& buffers) override;
    virtual bool verifyOutput(const samplesCommon::BufferManager& buffers) override;
};



// For the special implementation of language
class CLIP_Text: public CLIP_Base{
public:
    CLIP_Text(const samplesCommon::OnnxSampleParams& params):CLIP_Base(params){}

    virtual bool build() override;
    virtual bool infer() override;
    virtual bool processInput(const samplesCommon::BufferManager& buffers) override;
    virtual bool verifyOutput(const samplesCommon::BufferManager& buffers) override;
};


#endif