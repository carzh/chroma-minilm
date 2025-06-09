import Foundation
import MetalPerformanceShadersGraph

guard let url = Bundle.main.url(forResource: "mpsgraph/20250609", withExtension: "mpsgraphpackage") else {
    fatalError("Model not found")
}

let executable = try MPSGraphExecutable(coreMLPackageAtURL: url, descriptor: nil)

let device = MTLCreateSystemDefaultDevice()!
let graph = MPSGraph()

// Example shape: [1, 3, 640, 640]
// [batch, channels, height, width]
let inputShape: [NSNumber] = [1, 3, 480, 640]
let dataCount = inputShape.map { $0.intValue }.reduce(1, *)
let inputData = [Float](repeating: 0.0, count: dataCount) // replace with actual image data

let buffer = device.makeBuffer(bytes: inputData,
                               length: inputData.count * MemoryLayout<Float>.size,
                               options: [])!

let descriptor = MPSNDArrayDescriptor(dataType: .float32, shape: inputShape)
let inputArray = MPSNDArray(device: device, descriptor: descriptor)
inputArray.writeBytes(buffer.contents(), strideBytes: nil)

let inputName = executable.inputTensorNames.first!
let outputName = executable.outputTensorNames.first! // or all outputs

let inputDict = [inputName: inputArray]

let commandQueue = device.makeCommandQueue()!

let results = try executable.run(
    with: commandQueue,
    inputs: inputDict,
    results: nil,
    executionDescriptor: nil
)

for (i, tensorData) in results.enumerated() {
    print("🔹 Output \(i)")
    let ndarray = tensorData.mpsndarray()
    let elementCount = ndarray.resourceSize()

    // Prepare destination buffer
    var outputBuffer = [Float](repeating: 0.0, count: elementCount)

    // Read the data from GPU into outputBuffer
    outputBuffer.withUnsafeMutableBytes { rawBuffer in
        ndarray.readBytes(rawBuffer.baseAddress!, strideBytes: nil)
    }

    print("📈 Output values: \(outputBuffer)")
    print("🧬 Data type: \(tensorData.dataType)")  // likely .float32
}
// let outputCount = outputArray.descriptor.shape.reduce(1) { $0 * $1.intValue }
// let outputPointer = UnsafeMutablePointer<Float>.allocate(capacity: outputCount)
// defer { outputPointer.deallocate() }

// outputArray.readBytes(outputPointer, strideBytes: nil)
// let outputFloats = Array(UnsafeBufferPointer(start: outputPointer, count: outputCount))
