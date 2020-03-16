import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import torch
import acap
import random
import pickle


class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator):
    def __init__(self, input_layers, stream):
        trt.IInt8EntropyCalibrator.__init__(self)
        self.input_layers = input_layers
        self.stream = stream

        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names='data'):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self, length):
        return None

    def write_calibration_cache(self, ptr, size):
        # cache = ctypes.c_char_p(int(ptr))
        # with open('calibration_cache.bin', 'wb') as f:
        #     f.write(cache.value)
        return None


class ImageBatchStream():
    def __init__(self, batch_size, calibration_files):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) // batch_size) + \
                           (1 if (len(calibration_files) % batch_size) else 0)
        self.final_shape = [448, 448]
        self.files = calibration_files
        self.calibration_data = np.zeros((batch_size, 3, self.final_shape[1], self.final_shape[0]), dtype=np.float32)
        self.batch = 0

        self.cap = acap.acap()
        self.frame = np.zeros((3, self.final_shape[0], self.final_shape[1]), dtype=np.float32)
        self.buf = np.zeros((3, 2000, 2000), dtype=np.int8)

    def read_image_chw(self, path):
        num = random.sample([0, 1, 2, 3], 1)[0]
        self.cap.decode(path, 3, num, self.final_shape[1], 1, self.frame.ctypes._data, self.buf.ctypes._data)
        clip = self.frame.copy()
        return clip

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch:self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                print("[ImageBatchStream] Processing ", f)
                img = self.read_image_chw(f)
                # img = self.preprocessor(img)
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


def main():
    root = '/home/share/aichallenge/'
    model_name = 'resnet-101-aic-448-9000'
    model_path = 'onnx/' + model_name + '.onnx'
    f = open(os.path.join(root, 'MyvalEncode30OneLabel.pickle'), 'rb')
    calibration_files = pickle.load(f)
    np.random.shuffle(calibration_files)
    calibration_files = calibration_files[:128]
    for i, item in enumerate(calibration_files):
        calibration_files[i] = item[0].replace('/home/kcheng/AItemp/data/', '/home/share/aichallenge/')
    f.close()

    batchstream = ImageBatchStream(32, calibration_files)
    int8_calibrator = PythonEntropyCalibrator(["data"], batchstream)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    builder.int8_mode = True
    builder.int8_calibrator = int8_calibrator
    builder.max_batch_size = 4
    builder.max_workspace_size = 10000000000

    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(model_path, 'rb') as model:
        parser.parse(model.read())

    # do calibration and optimization here
    engine = builder.build_cuda_engine(network)
    with open("engine/" + model_name + '-b32-f128-nettrt' + '.engine', "wb") as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    main()
