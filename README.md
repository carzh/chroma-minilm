# chroma-minilm

## repro steps
1. install reqs: `pip install -r requirements.txt`
2. run generate-tensor-inputs: `python scripts/generate-tensor-inputs.py`
3. clone onnxruntime github repo and run the following for each input: `python C:\Users\carolinezhu\ort\tools\python\onnx_test_data_utils.py --action numpy_to_pb --input np_inputs/token_type_ids.npy --name token_type_ids --output input_2.pb`
4. move pb files into a test directory
5. run the run_test_runner_tool.py script from root directory of this project
