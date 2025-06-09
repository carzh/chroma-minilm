import Foundation
import MetalPerformanceShadersGraph

// Performance tracking variables
var loadStartTime: CFTimeInterval = 0
var loadEndTime: CFTimeInterval = 0
var firstInferenceTime: CFTimeInterval = 0
var totalInferenceTime: CFTimeInterval = 0
var inferenceCount: Int = 0
var peakMemoryUsage: UInt64 = 0

// CPU usage tracking
class CPUUsageTracker {
    private var startCPUTime: clock_t = 0
    private var measurements: [Double] = []
    
    func startTracking() {
        startCPUTime = clock()
    }
    
    func recordMeasurement() {
        let currentTime = clock()
        let cpuTimeUsed = Double(currentTime - startCPUTime) / Double(CLOCKS_PER_SEC)
        let wallTimeElapsed = 1.0 // Assuming 1 second intervals for simplicity
        let cpuUsage = (cpuTimeUsed / wallTimeElapsed) * 100.0
        measurements.append(cpuUsage)
        startCPUTime = currentTime
    }
    
    func getAverageCPUUsage() -> Double {
        return measurements.isEmpty ? 0.0 : measurements.reduce(0, +) / Double(measurements.count)
    }
}

// Memory tracking function
func getCurrentMemoryUsage() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    
    if kerr == KERN_SUCCESS {
        return info.resident_size
    }
    return 0
}

// Initialize CPU usage tracker
let cpuTracker = CPUUsageTracker()
cpuTracker.startTracking()

// Start timing package load
loadStartTime = CFAbsoluteTimeGetCurrent()

guard let url = Bundle.main.url(forResource: "mpsgraph/20250609", withExtension: "mpsgraphpackage") else {
    fatalError("Model not found")
}

let descriptor = MPSGraphCompilationDescriptor()
let executable = try MPSGraphExecutable(package: url, descriptor: descriptor)

// End timing package load
loadEndTime = CFAbsoluteTimeGetCurrent()

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

let arrayDescriptor = MPSNDArrayDescriptor(dataType: .float32, shape: inputShape)
let inputArray = MPSNDArray(device: device, descriptor: arrayDescriptor)
inputArray.writeBytes(buffer.contents(), strideBytes: nil)

let inputTensor = MPSGraphTensorData(buffer, shape: inputShape, dataType: .float32)
let inputDict: [MPSGraphTensorData] = [inputTensor]

let commandQueue = device.makeCommandQueue()!

// Function to run inference with timing
func runInference() -> CFTimeInterval {
    let startTime = CFAbsoluteTimeGetCurrent()
    
    // Track memory before inference
    let currentMemory = getCurrentMemoryUsage()
    if currentMemory > peakMemoryUsage {
        peakMemoryUsage = currentMemory
    }
    
    let results = try! executable.run(
        with: commandQueue,
        inputs: inputDict,
        results: nil,
        executionDescriptor: nil
    )
    
    // Ensure GPU work is complete by creating and committing a command buffer
    let commandBuffer = commandQueue.makeCommandBuffer()!
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let endTime = CFAbsoluteTimeGetCurrent()
    let inferenceTime = endTime - startTime
    
    // Track memory after inference
    let postMemory = getCurrentMemoryUsage()
    if postMemory > peakMemoryUsage {
        peakMemoryUsage = postMemory
    }
    
    // Record CPU usage
    cpuTracker.recordMeasurement()
    
    return inferenceTime
}

// Run multiple inferences for better statistics
let numberOfInferences = 1000

print("🚀 Starting performance evaluation...")
print("📊 Running \(numberOfInferences) inferences...")

for i in 0..<numberOfInferences {
    let inferenceTime = runInference()
    
    if i == 0 {
        firstInferenceTime = inferenceTime
        print("✅ First inference completed")
    }
    
    totalInferenceTime += inferenceTime
    inferenceCount += 1
    
    // Small delay between inferences to get better CPU measurements
    // usleep(100000) // 100ms
}

// Calculate metrics
let packageLoadTime = loadEndTime - loadStartTime
let averageInferenceTime = totalInferenceTime / Double(inferenceCount)
let inferencesPerSecond = 1.0 / averageInferenceTime
let averageCPUUsage = cpuTracker.getAverageCPUUsage()
let peakMemoryMB = Double(peakMemoryUsage) / (1024.0 * 1024.0)

// Print results
print("\n" + String(repeating: "=", count: 60))
print("🎯 PERFORMANCE METRICS")
print(String(repeating: "=", count: 60))
print("📦 Package load time: \(String(format: "%.4f", packageLoadTime * 1000)) ms")
print("🏃 First inference time: \(String(format: "%.4f", firstInferenceTime * 1000)) ms")
print("⏱️  Total inference time: \(String(format: "%.4f", totalInferenceTime * 1000)) ms")
print("🔢 Total inference requests: \(inferenceCount)")
print("📊 Average inference time: \(String(format: "%.4f", averageInferenceTime * 1000)) ms")
print("⚡ Inferences per second: \(String(format: "%.2f", inferencesPerSecond))")
print("💻 Average CPU usage: \(String(format: "%.2f", averageCPUUsage))%")
print("🧠 Peak working set size: \(String(format: "%.2f", peakMemoryMB)) MB")
print(String(repeating: "=", count: 60))