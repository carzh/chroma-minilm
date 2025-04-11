import onnx  
model = onnx.load("../test_1/model.onnx")  
initializer_names = {init.name for init in model.graph.initializer}  
print(initializer_names)