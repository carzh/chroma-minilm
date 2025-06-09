Command for converting onnx model to mpsgraph package:

```bash
mpsgraphtool convert -onnx yolov8m.onnx -path mpsgraph -packagename [package name]
```

command for compiling swift script:
```bash
swiftc run.swift -o run_model -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
```


I got the yolo onnx model from the following colab:
https://colab.research.google.com/drive/1-yZg6hFg27uCPSycRCRtyezHhq_VAHxQ?usp=sharing#scrollTo=rKnUE62F925P

from this github: https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection?tab=readme-ov-file


command for compiling perf script: 
```bash
swiftc -o perf perf.swift -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
```

steps for running onnxruntime_perf_test for yolo:
1. build onnxruntime
2. run `python generate_npy_inputs.py`
3. run `python /Users/carolinezhu/Documents/onnxruntime/tools/python/onnx_test_data_utils.py --action numpy_to_pb --input images.npy --name images --output input_0.pb`
4. create the following folder structure:
```
yolov8/
└── test_1/
    ├── model.onnx
    └── test_input_1/
        └── input_0.pb
```