/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! clip_demo.cpp
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "clip.h"

const std::string gSampleName = "CLIP_demo";

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::VisionParams initializeVisionParams(const samplesCommon::Args& args)
{
    samplesCommon::VisionParams params;
    if(args.dataDirs.empty())
    {
        params.dataDirs.emplace_back("../../model/");
    }
    else{
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = args.vision_model_name.empty()? "v.onnx" : args.vision_model_name;
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.image_mean = {0.48145466, 0.4578275, 0.40821073};
    params.image_std = {0.26862954, 0.26130258, 0.27577711};   
    params.image_src = params.dataDirs.size()>1? params.dataDirs[1]:"/home/TensorRT-8.6.1.6/samples/my_demo/include/common";

    return params;
}

samplesCommon::LanguageParams initializeLanguageParams(const samplesCommon::Args& args)
{
    samplesCommon::LanguageParams params;

    if(args.dataDirs.empty())
    {
        params.dataDirs.emplace_back("../../model/");
    }
    else{
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = args.language_model_name.empty()? "t.onnx" : args.language_model_name;
    params.inputTensorNames.push_back("input");
    params.outputTensorNames.push_back("output");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.int8 = args.runInFp16;
    params.text_token_length = 77;
    params.vocab_path = "../../vocab//bpe_simple_vocab_16e6.txt";
    params.text_src = params.dataDirs.size()>2? params.dataDirs[2]: "123.txt";

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl<<std::endl;
    std::cout << "--help          Display help information" << std::endl<<std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. The first path corresponds to the folder where the model "
                 "is located If no data directories are given, the default is to use '(../../model/)'. The second path "
                 "(options) corresponds to the path of the image file or folder. The third path corresponds to the string "
                 "or text file (. txt) to be retrieved."
              << std::endl<<std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl<<std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl<<std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl<<std::endl;
    std::cout << "--vision_model_name       Specify the name of the visual model." << std::endl<<std::endl;
    std::cout << "--language_model_name     Specify the name of the language model." << std::endl<<std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    CLIP_Vision V_Model(initializeVisionParams(args));
    CLIP_Text L_Model(initializeLanguageParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for CLIP" << std::endl;

    if (!V_Model.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    // if (!L_Model.build())
    // {
    //     return sample::gLogger.reportFail(sampleTest);
    // }
    if (!V_Model.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    // if (!L_Model.infer())
    // {
    //     return sample::gLogger.reportFail(sampleTest);
    // }


    return sample::gLogger.reportPass(sampleTest);
}
