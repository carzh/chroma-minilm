import Foundation
import Metal
import MetalPerformanceShadersGraph

// swiftc test_run.swift -o swift_run_model -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph

func runMPSGraphModel(
    at modelPath: String,
    batchSize: Int,
    sequenceLength: Int
) {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("❌ Metal is not supported on this device.")
        return
    }

    let modelURL = URL(fileURLWithPath: modelPath)
    
    do {
        let executable = try MPSGraphExecutable(
            coreMLPackageAtURL: modelURL,
            descriptor: nil
        )

        // Prepare dummy input data
        let inputCount = batchSize * sequenceLength

        let commandQueue = device.makeCommandQueue()!

        let inputIDs: [Int32] = Array(repeating: 1, count: inputCount)
        let attentionMask: [Int32] = Array(repeating: 1, count: inputCount)
        let tokenTypeIDs: [Int32] = Array(repeating: 0, count: inputCount)

        func createTensor(from data: [Int32]) -> MPSGraphTensorData {
            let length = data.count * MemoryLayout<Int32>.stride
            let buffer = device.makeBuffer(bytes: data, length: length, options: .storageModeShared)!
            return MPSGraphTensorData(
                buffer,
                shape: [NSNumber(value: batchSize), NSNumber(value: sequenceLength)],
                dataType: .int32
            )
        }

        // Reorder inputs to match executable.inputNames order
        let orderedInputs: [MPSGraphTensorData] = [
            createTensor(from: inputIDs),        // input_ids
            createTensor(from: attentionMask),  // attention_mask
            createTensor(from: tokenTypeIDs)    // token_type_ids
        ]

        let outputs = try executable.run(
            with: commandQueue,
            inputs: orderedInputs,
            results: nil,
            executionDescriptor: nil
        )

        // Read data from MPSNDArray inside the MPSGraphTensorData

        for (i, tensorData) in outputs.enumerated() {
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
    } catch {
        print("❌ Error loading or running model: \(error)")
    }
}

runMPSGraphModel(
    at: "output.mpsgraphpackage",
    batchSize: 1,
    sequenceLength: 12
)
