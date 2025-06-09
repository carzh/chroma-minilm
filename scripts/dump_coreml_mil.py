import argparse
import coremltools.proto.Model_pb2 as Model_pb2

def inspect_mlmodel(mlmodel_path):
    """Parse Core ML .mlmodel file using embedded protobuf definitions"""
    with open(mlmodel_path, "rb") as f:
        model = Model_pb2.Model()
        model.ParseFromString(f.read())
        
    # Print basic structure
    print(f"Model Type: {model.WhichOneof('Type')}")
    print(f"Spec Version: {model.specificationVersion}")
    
    # Inspect neural network layers (if present)
    if model.HasField("neuralNetwork"):
        for layer in model.neuralNetwork.layers:
            print(f"\nLayer: {layer.name}")
            print(f"  Type: {layer.WhichOneof('layer')}")
            print(layer)
            
    # For ML Program models
    elif model.HasField("mlProgram"):
        print("\nML Program Details:")
        main_func = model.mlProgram.functions["main"]
        print(main_func.block_specializations)
        # operations = main_func.block_specializations.operations  # Correct access
        # print(f"Operations count: {len(operations)}")
        print(f"Inputs: {[i.name for i in main_func.inputs]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect CoreML model')
    parser.add_argument('model_path', type=str, help='Path to .mlmodel file')
    args = parser.parse_args()
    
    inspect_mlmodel(args.model_path)