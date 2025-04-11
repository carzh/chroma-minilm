import argparse
import os

import onnx
from onnx import shape_inference


def update_model(model_path):
    pieces = os.path.splitext(model_path)
    new_path = pieces[0] + '.si' + pieces[1]

    m = onnx.load_model(model_path)
    inferred_model = shape_inference.infer_shapes(m)

    onnx.save_model(inferred_model, new_path)


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('model', help="model file. updated model will be written to '.si.onnx'")
  return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    update_model(args.model)
