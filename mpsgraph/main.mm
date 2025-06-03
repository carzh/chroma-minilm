#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <iostream>

// build with:
// clang++ -std=c++17 main.mm -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph -o run_model

void runMPSGraphPackage(const char* pathToPackage) {
    @autoreleasepool {
        // 1. Create a Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device.\n";
            return;
        }

        // 2. Load the .mpsgraphpackage file
        NSURL *packageURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:pathToPackage]];
        NSError *error = nil;

        MPSGraphExecutable *executable = [MPSGraphExecutable graphExecutableWithContentsOfURL:packageURL
                                                                                       device:device
                                                                                      options:nil
                                                                                        error:&error];

        if (error || !executable) {
            std::cerr << "Failed to load graph package: " << [[error localizedDescription] UTF8String] << "\n";
            return;
        }

        // 3. Create a command queue
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // 4. Create an input buffer (example: 1x4 float tensor)
        const int inputElementCount = 4;
        float inputValues[inputElementCount] = {1.0f, 2.0f, 3.0f, 4.0f};

        NSUInteger bufferSize = sizeof(inputValues);
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputValues
                                                        length:bufferSize
                                                       options:MTLResourceStorageModeShared];

        MPSGraphTensorData *inputTensor = [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuffer
                                                                                   shape:@[@1, @4]
                                                                                 dataType:MPSDataTypeFloat32];

        // 5. Set up feeds
        NSDictionary<NSString*, MPSGraphTensorData*> *feeds = @{
            @"input": inputTensor // assumes the model's input tensor is named "input"
        };

        // 6. Run the model and fetch "output"
        NSDictionary<NSString*, MPSGraphTensorData*> *results = [executable runWithFeeds:feeds
                                                                             targetTensors:@[@"output"]
                                                                                   options:nil
                                                                                    error:&error];
        if (error) {
            std::cerr << "Error during graph execution: " << [[error localizedDescription] UTF8String] << "\n";
            return;
        }

        MPSGraphTensorData *outputTensor = results[@"output"];

        NSData *outputData = [outputTensor data];
const float *outputValues = (const float *)[outputData bytes];


        std::cout << "Model output:\n";
        for (int i = 0; i < inputElementCount; ++i) {
            std::cout << "  " << outputValues[i] << "\n";
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " path/to/model.mpsgraphpackage\n";
        return 1;
    }

    runMPSGraphPackage(argv[1]);
    return 0;
}
