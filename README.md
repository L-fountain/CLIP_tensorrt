# How to use

## Step 1 Convert model
1. Obtain the [CLIP](https://github.com/openai/CLIP) model

2. Use [clip2onnx](https://github.com/L-fountain/clip2onnx) to convert it from pt to onnx

3. (**Options**) [PTQ](https://github.com/L-fountain/clip2onnx/deploy/int8) quantification model 


## Step2 Preprocess
### Two ways to implement image preprocess
1. [Using cuda preprocess](https://blog.csdn.net/zi_y_uan/article/details/130932829)

2. [Using Opencv preprocess](https://blog.csdn.net/weixin_45137428/article/details/122229184)

    **By the way:** Multiple batch preprocessing can refer to [this](https://github.com/monatis/clip.cpp/blob/main/clip.cpp)

### The way to tokenize the words
1. You can refer to [clip_tokenizer](https://github.com/ozanarmagan/clip_tokenizer_cpp)


## Step3 Build
1. Dynamic batchsize code like [reference_1](https://zhuanlan.zhihu.com/p/392345898) 

    **Note**: Please refer to [reference_2](https://blog.csdn.net/XCCCCZ/article/details/122990377) for the solution to the issue where the BufferManager of TensorRT samples does not support engines with dynamic dimensions


## Step4 Infer


## Step5 Compile
1. Modify and correctly configure dependency paths in CmakeLists at all levels of directories

    **Note**: The code uses the **filesystem** library of C++17. If C++17 is not supported, the corresponding functions and modules need to be downgraded and replaced

2. Perform the following actions in the root directory of the project
    ```
        mkdir build
        cd build
        cmake ..
        make
    ```

3. The executable file will be generated in the build/bin directory, and specific runtime parameters can refer to [main.cpp](https://github.com/L-fountain/CLIP_tensorrt/blob/main/src/main.cpp) and [argsParser.h](https://github.com/L-fountain/CLIP_tensorrt/blob/main/include/common/argsParser.h)

	