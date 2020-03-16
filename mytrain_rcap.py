import os
import mxnet as mx
from mxnet.gluon.data import DataLoader, Dataset
import torch
import ncap, ncap3d, acap, rcap
import pickle
import random
from numpy.random import randint as rint
import numpy as np
import logging
from symbol_resnet import resnet
import argparse

head = '%(asctime)-15s %(message)s'
# MXNET_CUDNN_AUTOTUNE_DEFAULT=0
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
parser = argparse.ArgumentParser(description="command for training resnet-v2")
parser.add_argument('--num_gpus', type=int, default='4', help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--model', type=str, default='resnet-101-aic-001-448-119-yuv')
parser.add_argument('--lr', type=float, default=0.001, help='initialization learning reate')
parser.add_argument('--batch-size', type=int, default=32, help='the batch size')
parser.add_argument('--num-classes', type=int, default=119, help='the class number of your task')
parser.add_argument('--frequent', type=int, default=100, help='frequency of logging')
parser.add_argument('--epoch', type=int, default=40, help='frequency of logging')
parser.add_argument('--epoch_start', type=int, default=0, help='frequency of logging')
parser.add_argument('--img_size', type=int, default=448, help='frequency of logging')

args = parser.parse_args()
# logging.info(args)
logging.basicConfig(level=logging.DEBUG, format=head)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/' + args.model + '.log', mode='w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.info(args)


class MeituImgFolder(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.final_shape = [args.img_size, args.img_size]
        root = '/home/share/aichallenge/'

        if mode == 'train':
            f = pickle.load(open(os.path.join(root, 'MytrianEncode30OneLabel.pickle'), 'rb'))
            self.clips = f
        elif mode == 'val':
            f = open(os.path.join(root, 'MyvalEncode30OneLabel.pickle'), 'rb')
            self.clips = pickle.load(f)

        self.cap = rcap.rcap()
        self.y = np.zeros((2000 * 2000 * 3), dtype=np.uint8)
        self.u = np.zeros((1000 * 1000 * 3), dtype=np.uint8)
        self.v = np.zeros((1000 * 1000 * 3), dtype=np.uint8)
        self.size = np.zeros((2), np.float32)
        self.frame = np.zeros((3, self.final_shape[0], self.final_shape[1]), dtype=np.float32)

        print('length of ', root, '   ', len(self.clips))

    def __getitem__(self, index):
        paths_total, label = self.clips[index]
        paths_total = paths_total.replace('/home/kcheng/AItemp/data/', '/home/share/aichallenge/')

        num = random.sample(list(range(4)), 1)[0]

        self.cap.decode(paths_total, 4, num, self.final_shape[1], 1, self.y.ctypes._data, self.u.ctypes._data,
                        self.v.ctypes._data, self.size.ctypes._data)
        if self.size[0] < self.final_shape[0]:
            self.size[0] = self.final_shape[0]
        if self.size[1] < self.final_shape[0]:
            self.size[1] = self.final_shape[0]
        b = self.y[:int(self.size[0] * self.size[1])].reshape((int(self.size[0]), int(self.size[1])))
        c = self.u[:int(self.size[0] * self.size[1] / 4)].reshape((int(self.size[0] / 2), int(self.size[1] / 2)))
        d = self.v[:int(self.size[0] * self.size[1] / 4)].reshape((int(self.size[0] / 2), int(self.size[1] / 2)))
        gap_h = int((self.size[0] - self.final_shape[0]) / 2)
        gap_w = int((self.size[1] - self.final_shape[1]) / 2)
        self.frame[0, :, :] = b[gap_h:gap_h + self.final_shape[1], gap_w:gap_w + self.final_shape[0]].astype(np.float32)
        self.frame[1, ::2, ::2] = self.frame[1, 1::2, 1::2] \
            = self.frame[1, 0::2, 1::2] = self.frame[1, 1::2, 0::2] = \
            c[gap_h // 2:gap_h // 2 + self.final_shape[0] // 2,
            gap_w // 2:gap_w // 2 + self.final_shape[1] // 2].astype(np.float32)
        self.frame[2, ::2, ::2] = self.frame[2, 1::2, 1::2] \
            = self.frame[2, 0::2, 1::2] = self.frame[2, 1::2, 0::2] = \
            d[gap_h // 2:gap_h // 2 + self.final_shape[0] // 2,
            gap_w // 2:gap_w // 2 + self.final_shape[1] // 2].astype(np.float32)

        return self.frame, np.float32(label)

    def __len__(self):
        return len(self.clips)


data_set_train = MeituImgFolder('train')
data_set_val = MeituImgFolder('val')
data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=8, last_batch='discard', pin_memory=True)
data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=16, last_batch='discard', pin_memory=True, )


class SimpleIter(object):
    def __init__(self, gluon_data_loader):
        self.gluon_data_loader = gluon_data_loader
        self.gluon_data_loader_iter = iter(self.gluon_data_loader)

        data, label = next(self.gluon_data_loader_iter)
        data_desc = mx.io.DataDesc(name='data', shape=data.shape, dtype=data.dtype)
        label_desc = mx.io.DataDesc(name='softmax_label', shape=label.shape, dtype=label.dtype)

        self.gluon_data_loader_iter = iter(self.gluon_data_loader)

        self._provide_data = [data_desc]
        self._provide_label = [label_desc]

    def __iter__(self):
        return self

    def reset(self):
        self.gluon_data_loader_iter = iter(self.gluon_data_loader)

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        data, label = next(self.gluon_data_loader_iter)
        batch = mx.io.DataBatch(data=[data], label=[label], provide_data=self._provide_data,
                                provide_label=self._provide_label)
        return batch


train_iter = SimpleIter(data_loader_train)
val_iter = SimpleIter(data_loader_val)


# Model


def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='flatten0'):
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    print('get finetune model')
    all_layers = symbol.get_internals()
    net = all_layers[layer_name + '_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k: arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

model_prefix = './model/' + args.model
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.epoch_start)
# (new_sym, new_args) = get_fine_tune_model(sym, arg_params, args.num_classes)
new_sym, new_args = sym, arg_params


def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
            num_epoch=args.epoch,
            arg_params=arg_params,
            aux_params=aux_params,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(batch_size, args.frequent),
            epoch_end_callback=mx.callback.do_checkpoint(model_prefix),
            eval_end_callback=mx.callback.LogValidationMetricsCallback(),
            kvstore='device',
            optimizer='sgd',
            optimizer_params={'learning_rate': args.lr},
            initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
            eval_metric='acc',
            # begin_epoch=args.epoch_start+1
            )
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)


mod_score = fit(new_sym, new_args, aux_params, train_iter, val_iter, args.batch_size, args.num_gpus)
