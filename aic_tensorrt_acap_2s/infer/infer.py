# -*- coding: utf-8 -*-
from collections import namedtuple
import torch
import tensorrt as trt
import acap3d, acap
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os


class ServerApi(object):
    def __init__(self, gpu_id=0):
        self.d = np.load('../dictlshi.npy').tolist()
        self.dtype = np.float32
        self.final_shape = (4, 448, 448)
        self.batch_size = 4

        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        # resnet-101-aic-448-9000-b32-f640 res101-encode30-pytorch-b32-f640
        # f = open('/data/resnet-101-aic-448-9190-b4-b32-f128.engine', 'rb')
        f1 = open('../resnet-101-aic-448-acapfixall-9556-b32-f128-max2.engine', 'rb')
        self.engine1 = runtime.deserialize_cuda_engine(f1.read())
        f2 = open('../resnet-101-aic-448-acapfixall-9556-b32-f128-max2.engine', 'rb')
        self.engine2 = runtime.deserialize_cuda_engine(f2.read())
        f3 = open('../resnet-101-aic-448-acapfixall-9538-b32-f128.engine', 'rb')
        self.engine3 = runtime.deserialize_cuda_engine(f3.read())
        f4 = open('../resnet-101-aic-448-acapfixall-9556-b32-f128.engine', 'rb')
        self.engine4 = runtime.deserialize_cuda_engine(f4.read())

        self.input_shape1 = [self.batch_size, *self.engine1.get_binding_shape(0)]
        self.output_shape1 = [self.batch_size, *self.engine1.get_binding_shape(1)]
        self.output_shape2 = [self.batch_size, *self.engine2.get_binding_shape(1)]
        self.output_shape3 = [self.batch_size, *self.engine3.get_binding_shape(1)]
        self.output_shape4 = [self.batch_size, *self.engine4.get_binding_shape(1)]
        print(self.output_shape2)
        print(self.output_shape1)
        print(self.input_shape1)
        self.h_input1 = cuda.pagelocked_empty(self.input_shape1, dtype=np.float32)
        self.h_output1 = cuda.pagelocked_empty(self.output_shape1, dtype=np.float32)
        self.h_output2 = cuda.pagelocked_empty(self.output_shape2, dtype=np.float32)
        self.h_output3 = cuda.pagelocked_empty(self.output_shape3, dtype=np.float32)
        self.h_output4 = cuda.pagelocked_empty(self.output_shape4, dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        self.d_input1 = cuda.mem_alloc(self.h_input1.nbytes)
        self.d_output1 = cuda.mem_alloc(self.h_output1.nbytes)
        self.d_output2 = cuda.mem_alloc(self.h_output2.nbytes)
        self.d_output3 = cuda.mem_alloc(self.h_output3.nbytes)
        self.d_output4 = cuda.mem_alloc(self.h_output4.nbytes)
        # Create a self.stream in which to copy inputs/outputs and run inference.
        self.stream1 = cuda.Stream()
        self.stream2 = cuda.Stream()
        self.stream3 = cuda.Stream()
        self.stream4 = cuda.Stream()
        # self.stream = trt.
        self.context1 = self.engine1.create_execution_context()
        self.context2 = self.engine2.create_execution_context()
        self.context3 = self.engine3.create_execution_context()
        self.context4 = self.engine4.create_execution_context()

        self.buf = np.zeros((self.final_shape[0], 2000, 2000, 3), dtype=np.uint8)
        # self.clip = np.zeros((self.final_shape[0], 3, self.resize_shape[0], self.resize_shape[1]), dtype=self.dtype)

        self.cap = acap3d.acap3d()
        # videolist = ['/data/231125424.mp4', '/data/963193352.mp4']
        # for i in range(50):
        #     video = videolist[i%2]
        #     self.cap.decode(video, self.final_shape[0], self.final_shape[1], 1, self.h_input.ctypes._data,
        #                     self.buf.ctypes._data)

    def handle(self, video_dir):
        # print('before decode')
        self.cap.decode(video_dir, self.final_shape[0], self.final_shape[1], 1, self.h_input1.ctypes._data, self.buf.ctypes._data)
        # print('after decode')
        cuda.memcpy_htod_async(self.d_input1, self.h_input1, self.stream1)
        # Run inference.
        # print('before exec')
        self.context1.execute(batch_size=2, bindings=[int(self.d_input1), int(self.d_output1)])
        self.context2.execute(batch_size=2, bindings=[int(self.d_input1), int(self.d_output2)])
        # self.context1.execute_async(batch_size=4, bindings=[int(self.d_input1), int(self.d_output1)],
        #                            stream_handle=self.stream1.handle)

        # self.context2.execute_async(batch_size=2, bindings=[int(self.d_input1), int(self.d_output2)],
        #                            stream_handle=self.stream2.handle)
        #
        # self.context3.execute_async(batch_size=1, bindings=[int(self.d_input1), int(self.d_output3)],
        #                             stream_handle=self.stream3.handle)
        #
        # self.context4.execute_async(batch_size=1, bindings=[int(self.d_input1), int(self.d_output4)],
        #                             stream_handle=self.stream4.handle)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(self.h_output1, self.d_output1, self.stream1)
        cuda.memcpy_dtoh_async(self.h_output2, self.d_output2, self.stream2)
        # cuda.memcpy_dtoh_async(self.h_output3, self.d_output3, self.stream3)
        # cuda.memcpy_dtoh_async(self.h_output4, self.d_output4, self.stream4)
        # Synchronize the self.stream
        self.stream1.synchronize()
        self.stream2.synchronize()
        # self.stream3.synchronize()
        # self.stream4.synchronize()
        # print(self.h_output2.shape)
        prob1 = self.h_output1.mean(0).mean(-1).mean(-1)
        prob2 = self.h_output2.mean(0).mean(-1).mean(-1)
        # prob3 = self.h_output3.mean(0).mean(-1).mean(-1)
        # prob4 = self.h_output4.mean(0).mean(-1).mean(-1)
        # prob = prob1 + prob2+ prob3+ prob4
        prob = prob1 + prob2
        # prob = prob1
        pred = np.argsort(prob)[::-1][0]
        print(pred)
        # if isinstance(self.d[pred], tuple):
        #     res = list(self.d[pred])
        # else:
        #     res = list([self.d[pred]])
        # print(res)
        res = self.d[pred]
        return res


if __name__ == '__main__':
    import line_profiler

    file = '../231125424.mp4'
    s = ServerApi()
    # s.handle('../231125424.mp4')
    profile = line_profiler.LineProfiler(s.handle)
    profile.run('s.handle(file)')
    profile.print_stats()
