import os
import mxnet as mx
from mxnet.gluon.data import DataLoader, Dataset
import torch
import ncap, ncap3d, acap
import pickle
import random
from numpy.random import randint as rint
import numpy as np
import logging
from symbol_resnet import resnet
import argparse
from crossentropy import *

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
parser = argparse.ArgumentParser(description="command for training resnet-v2")
parser.add_argument('--num_gpus', type=int, default='1', help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--model', type=str, default='resnet-101-aic-01-448-acapfixmore-mullabel')
parser.add_argument('--lr', type=float, default=0.001, help='initialization learning reate')
parser.add_argument('--batch-size', type=int, default=1, help='the batch size')
parser.add_argument('--num-classes', type=int, default=63, help='the class number of your task')
parser.add_argument('--th', type=float, default=0.5, help='frequency of logging')
parser.add_argument('--epoch', type=int, default=30, help='frequency of logging')
parser.add_argument('--epoch_start', type=int, default=30, help='frequency of logging')
parser.add_argument('--img_size', type=int, default=448, help='frequency of logging')

args = parser.parse_args()


def fix_img(cap, frame, path, resize_shape, final_shape, mean, use_flip=False):
    num = random.sample(list(range(4)), 1)[0]
    cap.decode(path, 3, num, final_shape[1], 1, frame.ctypes._data)

    flip_rand = rint(0, 1)
    if flip_rand == 1 and use_flip:
        clip = np.flip(frame, 2).copy()
    else:
        clip = frame.copy()
    return clip


class MeituImgFolder(Dataset):
    def __init__(self, mode):
        self.mode = mode
        if args.img_size == 224:
            self.resize_shape = [240, 240]
            self.final_shape = [224, 224]
        else:
            self.resize_shape = [480, 480]
            self.final_shape = [448, 448]
        self.mean = np.zeros((3, self.resize_shape[0], self.resize_shape[1])).astype(np.float32)  # BGR
        self.mean[0].fill(0.5354)
        self.mean[1].fill(0.4942)
        self.mean[2].fill(0.4761)

        # self.final_ratios = [0.8, 0.8]
        root = '/home/share/aichallenge/'

        if mode == 'train':
            f = open(os.path.join(root, 'My_train_allLabel_list_3labelTrue.pickle'), 'rb')
            clip1 = pickle.load(f)
            f.close()
            f = open(os.path.join(root, 'My_val_allLabel_list_3labelTrue.pickle'), 'rb')
            clip2 = pickle.load(f)[:-1000]
            f.close()
            self.clips = clip1 + clip2

        elif mode == 'val':
            f = open(os.path.join(root, 'My_val_allLabel_list_3labelTrue.pickle'), 'rb')

            self.clips = pickle.load(f)[-1000:]

        else:
            raise (RuntimeError("mode not right"))

        self.cap = acap.acap()
        self.frame = np.zeros((3, self.final_shape[0], self.final_shape[1]), dtype=np.float32)

        print('length of ', root, '   ', len(self.clips))

    def __getitem__(self, index):
        paths_total, label = self.clips[index]
        paths_total = paths_total.replace('/home/kcheng/AItemp/data/', '/home/share/aichallenge/')
        clips = fix_img(self.cap, self.frame, paths_total, self.resize_shape, self.final_shape, self.mean)
        # for i, l in enumerate(label):
        #     label[i] = np.float32(l)
        l = np.zeros(args.num_classes)
        for i, num in enumerate(l.tolist()):
            if i in label:
                l[i] = 1
        return clips, l

    def __len__(self):
        return len(self.clips)


# Data


data_set_val = MeituImgFolder('val')
data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, last_batch='keep', pin_memory=True)


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


val_iter = SimpleIter(data_loader_val)


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

model_prefix = './model/' + args.model
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.epoch_start)


def acc(label, pred, label_width=args.num_classes):
    # pre = pred.tolist()
    # for i_b, b in enumerate(pre):
    #     for i_n, num in enumerate(b):
    #         if num > args.th:
    #             pre[i_b][i_n] = 1
    #         else:
    #             pre[i_b][i_n] = 0
    #     if sum(b)==0:
    #         pre[i_b][np.argmax(b)]=1

    # pre = np.round(pred).tolist()

    pre = pred.tolist()
    for i_b, b in enumerate(pre):
        max_b = max(b)
        th = max_b/1.1
        for i_n, num in enumerate(b):
            if num > th:
                pre[i_b][i_n] = 1
            else:
                pre[i_b][i_n] = 0

    batch = 0
    acc = 0
    for i_b, b in enumerate(pre):
        right = 0
        total = 0
        batch += 1
        for i_n, num in enumerate(b):
            if label[i_b, i_n] == 1 or num == 1:
                total += 1
            if 1 == label[i_b, i_n] and num == 1:
                right += 1
        acc += right / total
    acc /= batch

    return acc


def loss(label, pred):
    loss_all = 0
    for i in range(len(pred)):
        loss = 0
        loss -= label[i] * np.log(pred[i] + 1e-6) + (1. - label[i]) * np.log(1. + 1e-6 - pred[i])
        loss_all += np.sum(loss)
    loss_all = float(loss_all) / float(len(pred) + 0.000001)
    return loss_all


eval_metric = list()
eval_metric.append(mx.metric.np(acc))
eval_metric.append(mx.metric.np(loss))

devs = [mx.gpu(i) for i in range(args.num_gpus)]
mod = mx.mod.Module(symbol=sym, context=devs)
mod.bind(for_training=False,
         data_shapes=val_iter.provide_data,
         label_shapes=val_iter.provide_label)
mod.set_params(arg_params, aux_params)
mod_score = mod.score(val_iter, eval_metric)
print(mod_score)
