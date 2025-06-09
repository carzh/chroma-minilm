import argparse
import coremltools as ct
import numpy as np

def create_input_and_run_mlpackage(mlpackage_path):
    model = ct.models.MLModel(mlpackage_path)

    print(model.input_description)

    # unk__0, unk__1
    embeddings_slice_dummy_input = np.random.rand(2, 3).astype(np.int32)
    # unk__2, unk__3, 384
    embeddings_add_dummy_input = np.random.rand(4, 5, 384).astype(np.float32)
    # unk__5, unk__6, unk__7, unk__8
    expand_dummy_input = np.random.rand(2, 3, 4, 5).astype(np.int32)

    coreml_input = {"_embeddings_Slice_output_0": embeddings_slice_dummy_input,
                    "_embeddings_Add_output_0": embeddings_add_dummy_input,
                    "_Expand_output_0": expand_dummy_input}

    output = model.predict(coreml_input)
    print(output)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect CoreML model')
    parser.add_argument('model_path', type=str, help='Path to .mlmodel file')
    args = parser.parse_args()
    
    create_input_and_run_mlpackage(args.model_path)