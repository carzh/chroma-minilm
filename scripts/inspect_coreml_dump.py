import argparse
import coremltools as ct
# from coremltools.models.utils import get_weights_metadata

def inspect_coreml_model(model_path):
    try:
        # Load model and get spec
        model = ct.models.MLModel(model_path)
        spec = model.get_spec()
        
        # Print basic model info
        print(f"\nModel: {model_path}")
        print(f"Model Type: {spec.WhichOneof('Type')}")
        print(f"Spec Version: {spec.specificationVersion}")
        
        # Print input/output descriptions
        # print("\nInputs:")
        # for inp in spec.description.input:
        #     print(f"  {inp.name}: {inp.type.WhichOneof('Type')} {dict(inp.type.__getattribute__(inp.type.WhichOneof('Type')))}")
            
        # print("\nOutputs:")
        # for out in spec.description.output:
        #     print(f"  {out.name}: {out.type.WhichOneof('Type')} {dict(out.type.__getattribute__(out.type.WhichOneof('Type')))}")

        print("=" * 50)
        print(spec)
        
        # # Print weights metadata
        # weights_info = get_weights_metadata(spec)
        # print("\nWeight Layers:")
        # for layer_name, meta in weights_info.items():
        #     print(f"  {layer_name}:")
        #     print(f"    Shape: {meta['shape']}")
        #     print(f"    DataType: {meta['dataType']}")
        #     print(f"    Elements: {meta['numpy_array'].size}")
            
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect CoreML model')
    parser.add_argument('model_path', type=str, help='Path to .mlmodel file')
    args = parser.parse_args()
    
    inspect_coreml_model(args.model_path)