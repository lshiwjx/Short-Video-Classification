import tensorrt as trt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



TRT_LOGGER = trt.Logger(trt.Logger.INFO)

model_name = 'resnet-101-aic-448-9000-b1'
model_path = 'onnx/' + model_name + '.onnx'
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network()
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(model_path, 'rb') as model:
    parser.parse(model.read())

builder.max_batch_size = 1
# builder.max_workspace_size = 20<<1
builder.max_workspace_size = 10000000000
# builder.fp16_mode=True
engine = builder.build_cuda_engine(network)

with open("engine/" + model_name + '.engine', "wb") as f:
    f.write(engine.serialize())