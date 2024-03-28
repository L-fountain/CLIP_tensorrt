#include "clip.h"
#include "preprocess.h"

using namespace nvinfer1;

bool CLIP_Text::read_text(const std::string& text_src){
    std::filesystem::path filePath(text_src);
    if (filePath.extension() == ".txt")
    {
        std::ifstream infile;
        infile.open(text_src);
        if (!infile.good())
        {
            sample::gLogInfo << "can't open " << text_src << std::endl;
            return -1;
        }

        std::string s;
        while (getline(infile, s))
        {
            texts.push_back(s);
        }
        infile.close();
    }
    else
    {
        texts.push_back(text_src);
    }
}

bool CLIP_Text::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    // 针对动态batchsize, 创建优化配置文件, 文本的输入为batchsize*77
    auto profile = builder->createOptimizationProfile();
    ITensor *input = network->getInput(0);
    Dims dims = input->getDimensions();

    // 设置优化配置文件的尺寸
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims2{1, dims.d[1]});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims2{1, dims.d[1]});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims2{16, dims.d[1]});
    
    // 将优化配置文件添加到BuilderConfig中
    config->addOptimizationProfile(profile);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    // 判断输入输出是否符合
    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 2);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    return true;
}



bool CLIP_Text::infer()
{
    const int textLength = mInputDims.d[1];
    if(textLength != text_token_length)
    {
        sample::gLogInfo << "The configured text_token_length does not match the actual model" << std::endl;
        return false;
    }

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    
    int BATCH_SIZE = 3;
    Dims2 input_dims{BATCH_SIZE, textLength};  //Batchsize*textLength
    auto flag = context->setInputShape(mEngine->getIOTensorName(0), input_dims);

    if(!flag)
    {
       return false;
    }

    // Create RAII buffer manager object
    // 简单的说，RAII 的做法是使用一个对象，在其构造时获取资源，在对象生命期控制对资源的访问使之始终保持有效，最后在对象析构的时候释放资源
    samplesCommon::BufferManager buffers(mEngine, 0, context.get());
    auto start = std::chrono::high_resolution_clock::now();

    if (!processInput(buffers))
    {
        return false;
    }

    // tokenizer 单个短语 100层循环0.0303437 s左右 
    // 3个短语100层循环0.07秒左右(算内存拷贝)
    // for(int i=0; i<100; i++){
    // // Read the input data into the managed buffers
    // // ASSERT(mParams.inputTensorNames.size() == 1);
    //     if (!processInput(buffers))
    //     {
    //         return false;
    //     }
    // }

    auto end = std::chrono::high_resolution_clock::now();
    // 计算并输出时间差
    std::chrono::duration<double> elapsed_seconds = end - start;
    // 输出以秒为单位的时间差
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds.\n";

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    // data() C++11 new feature, returns a pointer to the first element inside the container
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

bool CLIP_Text::processInput(const samplesCommon::BufferManager& buffers)
{
    const int textLength = mInputDims.d[1];

    std::vector<std::string> words={"a diagram", "a dog", "a cat"};
    
    TokenizerResult result = tokenizer.tokenize(words);

    std::cout << "Tokens: " << std::endl;
    std::cout<< result.tokens[0].size()<<std::endl;
    for(const auto& temp: result.tokens){
        for (const auto& token : temp) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    int* hostDataBuffer = static_cast<int*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

    // 验证是否成功分配内存
    if (!hostDataBuffer) {
        throw std::runtime_error("Failed to allocate memory for the buffer.");
    }

    for (size_t i = 0; i < result.tokens.size(); ++i) {
        std::copy(result.tokens[i].begin(), result.tokens[i].end(), hostDataBuffer + i * 77);
    }


    return true;
}



bool CLIP_Text::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];

    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    std::ofstream outputFile("output.txt", std::ios::out);

    // 将输出写入文本文件
    for (int i = 0; i < outputSize; ++i) {
        outputFile << output[i];
        // 如果不是最后一个元素，则添加分隔符
        if (i != outputSize - 1) {
            outputFile << ",";
        }
    }

    outputFile.close();

    return true;
}


