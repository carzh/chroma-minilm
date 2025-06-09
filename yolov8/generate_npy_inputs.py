import numpy as np
import os

def generate_exact_swift_input():
    """
    Generate numpy array that exactly matches the Swift input:
    let inputData = [Float](repeating: 0.0, count: dataCount)
    """
    
    # Swift input shape: [1, 3, 480, 640]
    # [batch, channels, height, width]
    input_shape = [1, 3, 480, 640]
    
    # Calculate total number of elements (matching Swift's dataCount)
    data_count = 1
    for dim in input_shape:
        data_count *= dim
    
    print(f"🎯 Input shape: {input_shape}")
    print(f"📊 Total elements (dataCount): {data_count:,}")
    
    # Create zeros array exactly like Swift: [Float](repeating: 0.0, count: dataCount)
    input_data = np.zeros(data_count, dtype=np.float32)
    
    # Reshape to match the expected dimensions
    input_data_shaped = input_data.reshape(input_shape)
    
    np.save('images', input_data_shaped)
    
    return input_data_shaped

if __name__ == "__main__":
    input_array = generate_exact_swift_input()
    