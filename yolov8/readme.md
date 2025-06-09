Command for converting onnx model to mpsgraph package:

```bash
mpsgraphtool convert -onnx yolov8m.onnx -path mpsgraph -packagename [package name]
```

command for compiling swift script:
```bash
swiftc run.swift -o run_model -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph
```