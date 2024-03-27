#include "clip.h"
#include "preprocess.h"

using namespace nvinfer1;


bool CLIP_Vision::build()
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

    // 针对动态batchsize, 创建优化配置文件, 图像的输入为batchsize*C*H*W
    auto profile = builder->createOptimizationProfile();
    ITensor *input = network->getInput(0);
    Dims dims = input->getDimensions();

    // 设置优化配置文件的尺寸
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{16, dims.d[1], dims.d[2], dims.d[3]});
    
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

    // 定义文件路径
    std::string enginePath = "/home/clip2onnx/deploy/int8/VmodelInt8.engine";

    // 打开文件并读取Engine数据
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) { /* 文件打开失败处理 */ }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    std::streampos fsize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 分配内存来存储Engine数据
    std::vector<char> trtModelStream(fsize);
    file.read(trtModelStream.data(), fsize);
    file.close();

    // 反序列化Engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(trtModelStream.data(), fsize), samplesCommon::InferDeleter());

    // mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
    //     mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    // 判断输入输出是否符合
    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

    // 加载均值和方差变为常量内存
    init_constants(image_mean, image_std);

    return true;
}


bool CLIP_Vision::infer()
{
    // Create RAII buffer manager object
    const int channel = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    
    int BATCH_SIZE = 1;
    Dims4 input_dims{BATCH_SIZE, channel, inputH, inputW};  //NCHW
    bool flag = context->setInputShape(mEngine->getIOTensorName(0), input_dims);

    if(!flag)
    {
        return false;
    }

    // Create RAII buffer manager object
    // 简单的说，RAII 的做法是使用一个对象，在其构造时获取资源，在对象生命期控制对资源的访问使之始终保持有效，最后在对象析构的时候释放资源
    samplesCommon::BufferManager buffers(mEngine, 0, context.get());
    auto start = std::chrono::system_clock::now();

    if (!processInput(buffers))
    {
        return false;
    }

    // for(int i=0; i<1000; i++){
    // // Read the input data into the managed buffers
    // // ASSERT(mParams.inputTensorNames.size() == 1);
    //     if (!processInput(buffers))
    //     {
    //         return false;
    //     }
    // }

    auto end = std::chrono::system_clock::now();
    // 计算并输出时间差
    std::chrono::duration<double> elapsed_seconds = end - start;
    // 输出以秒为单位的时间差
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds.\n";

    // Memcpy from host input buffers to device input buffers
    // If GPU preprocessing is used, not copy
    if(!USE_GPU_PREPROCESS)
    {
        buffers.copyInputToDevice();
    }

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



bool CLIP_Vision::processInput(const samplesCommon::BufferManager& buffers)
{
    const int channel = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // 读取图片
    cv::Mat origin_img;
    origin_img = cv::imread("/home/OIP-C.jpg", cv::IMREAD_COLOR);

    // 检查图片是否成功读取
    if (origin_img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    
    if(USE_GPU_PREPROCESS)
    {
        // GPU Preprocess
        preprocess(origin_img, inputH, inputW, static_cast<float*>(buffers.getDeviceBuffer(mParams.inputTensorNames[0])));
    }
    else{
        // CPU Preprocess
        float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));

        // 验证是否成功分配内存
        if (!hostDataBuffer) {
            throw std::runtime_error("Failed to allocate memory for the buffer.");
        }

        const float mean_[3] = {image_mean[0], image_mean[1], image_mean[2]};
        const float std_[3] = {image_std[0], image_std[1], image_std[2]};
        cv::Size size(inputW, inputH);
        StandNorm_c3(hostDataBuffer, origin_img, Norm::mean_std(mean_, std_, 1/255.0, ChannelType::Invert), size);
    }

    return true;
}


bool CLIP_Vision::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    std::cout << outputSize << std::endl;
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));

    std::ofstream outputFile("output.txt", std::ios::out);

    // 将输出写入文本文件
    for (int i = 0; i < 512; ++i) {
        outputFile << output[i];
        // 如果不是最后一个元素，则添加分隔符
        if (i != 512 - 1) {
            outputFile << ",";
        }
    }

    outputFile.close();

    return true;
}
