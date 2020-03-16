import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet

sym = 'model/resnet-101-aic-001-448-119-acapfixmore-symbol.json'
params = 'model/resnet-101-aic-001-448-119-acapfixmore-0020.params'
input_shape = (3, 3, 448, 448)
onnx_file = 'onnx/resnet-101-aic-448-9000-b3-e20.onnx'

converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)

# from onnx import checker
# import onnx
# # model_name = '3d_notOK'
# # converted_model_path = 'onnx/' + model_name + '.onnx'
# # Load onnx model
# model_proto = onnx.load(converted_model_path)
#
# # Check if converted ONNX protobuf is valid
# checker.check_graph(model_proto.graph)
