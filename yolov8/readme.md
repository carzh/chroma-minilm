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