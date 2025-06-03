import Metal
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let graph = MPSGraph(device: device)
try graph.load(from: "../output.mpsgraphpackage/model_0.mpsgraph")

/*
{'input_ids': array([[ 101, 2023, 2003, 2019, 2742, 6251,  102],
       [ 101, 2169, 6251, 2003, 4991,  102,    0]]), 
       
'token_type_ids': array([[0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0]]), 
       
'attention_mask': array([[1, 1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1, 0]])}
       */
let inputArray: [Int64] = [101, 2023, 2003, 2742, 6251, 102, 
                           101, 2169, 6251, 2003, 4991, 102, 0]

let inputBuffer = inputArray.asBuffer(on: device)!

let inputTensorData = MPSGraphTensorData(
    dataType: .int64,
    shape: [2, 7],
    buffer: inputBuffer
)

let tokenTypeArray: [Int64] = [0, 0, 0, 0, 0, 0, 0, 
                              0, 0, 0, 0, 0, 0, 0]

let tokenTypeBuffer = tokenTypeArray.asBuffer(on: device)!

let tokenTypeTensorData = MPSGraphTensorData(
    dataType: .int64,
    shape: [2, 7],
    buffer: tokenTypeBuffer
)

let attentionMaskArray: [Int64] = [1, 1, 1, 1, 1, 1, 1, 
                                  1, 1, 1, 1, 1, 1, 0]

let attentionMaskBuffer = attentionMaskArray.asBuffer(on: device)!

let attentionMaskData = MPSGraphTensorData(
    dataType: .int64,
    shape: [2, 7],
    buffer: attentionMaskBuffer
)

let commandBuffer = MPSCommandBuffer.makeCommandBuffer()
let outputs = graph.run(
    feeds: [
        "input:0": inputTensorData,
        "input:1": tokenTypeTensorData,
        "input:2": attentionMaskData
    ],
    targetTensors: ["output_0"]
) 

commandBuffer.commit()

let outputBuffer = outputs["output_0"]!.buffer
let outputArray = [Float32](buffer: outputBuffer)