import json
import os
import sys
import time
import numpy as np
import pickle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

DATA_DIR = os.path.join(ROOT_DIR, 'data')
# VIDEO_DIR = os.path.join(DATA_DIR, 'video')
VIDEO_DIR = '/home/share/aichallenge/val/'
# VIDEO_DIR = '/'

OUTPUT_DIR = os.path.join(ROOT_DIR, 'data')

from infer.infer import ServerApi


class VerifyServer(object):
    def __init__(self, gpu_id=0):
        self.server = ServerApi(gpu_id)

    def time_counter(self):
        """
        处理耗时统计
        :return:    平均耗时
        """
        N = 3  # 遍历次数
        num = 0  # 文本数
        total = 0.0  # 总耗时
        time_list = []
        video_list = self._file_read(os.path.join(DATA_DIR, 'input_time.txt'))
        # 循环预测时间列表
        for i in range(N):
            count, t_val, result_list = self._verify_method(video_list)
            num = count
            time_list.append(t_val)
            # 记录预测结果
            path = os.path.join(OUTPUT_DIR, 'output_time_%s.txt' % i)
            self._file_write(path, result_list)
        # 去除最大、最小值算平均速度
        time_list.sort()
        for item in time_list[1: -1]:
            total += item
        avg_time = total / (num * (N - 2))
        return avg_time

    def acc_counter(self):
        """
        准确率统计
        :return:    准确率
        """
        video_list = self._file_read(os.path.join(DATA_DIR, 'input_acc.txt'))
        label_list = self._file_read(os.path.join(DATA_DIR, 'tag.txt'), 'tag')
        # print(len(video_list), len(label_list))
        count, t_val, result_list = self._verify_method(video_list)
        # path = os.path.join(OUTPUT_DIR, 'output_acc.txt')
        # self._file_write(path, result_list)
        # 准确率统计
        ratio_list = []
        # f = open('../wrong_sample_4frame_res101.pickle', 'wb')
        # wrong_list = []
        for ind, val in enumerate(result_list):
            intersection_num = len(list(set(val[1]).intersection(set(label_list[ind]))))
            union_num = len(list(set(val[1]).union(set(label_list[ind]))))
            # if intersection_num/float(union_num)!=1:
                # s = str(video_list[ind]) + ',' + str(label_list[ind][0]) +'\n'
                # print((video_list[ind], int(label_list[ind][0])))
                # wrong_list.append((video_list[ind], int(label_list[ind][0])))
                # f.write(s)
            ratio_list.append(intersection_num / float(union_num))
        # pickle.dump(wrong_list, f)
        accuracy = np.mean(ratio_list)
        # 记录预测结果
        path = os.path.join(OUTPUT_DIR, 'output_acc.txt')
        self._file_write(path, result_list)
        avg_time = t_val / float(count)
        return accuracy, avg_time

    def _verify_method(self, video_list):
        """
        验证方法
        :param video_list: 视频列表
        :return:
        """
        t_start_time = time.time()
        # 视频处理
        result_list = []
        for video in video_list:
            start_time = time.time()
            video_name = video.split('/')[-1]
            try:
                cla_list = self.server.handle(video)
                cla_list = [str(i) for i in cla_list]
                if len(cla_list) == 0: cla_list = ['-1']
            except Exception:
                cla_list = ['-1']  # -1代表预测异常
            result_list.append([video_name, cla_list, time.time() - start_time])
        return len(video_list), time.time() - t_start_time, result_list

    def save(self, speed=0.0, accuracy=0.0, acc_time=0.0):
        result = {
            "time": speed,
            "acc_rate": accuracy,
            "acc_time": acc_time
        }
        with open(os.path.join(OUTPUT_DIR, 'result.txt'), 'w+') as fw:
            fw.write(json.dumps(result))

    def _file_read(self, path, type='video'):
        r_list = []
        with open(path, 'r+') as fr:
            line = fr.readline()
            while line:
                line = line.replace('\n', '')
                if 'tag' == type:
                    r_list.append(line.split(',')[1:])
                else:
                    r_list.append(os.path.join(VIDEO_DIR, line))
                line = fr.readline()
        return r_list

    def _file_write(self, path, val_list):
        with open(path, 'w+') as fw:
            for v in val_list:
                out_info = '%s,%s,%s' % (v[0], str(','.join(v[1])), str(v[2])) + '\n'
                fw.write(out_info)


if __name__ == '__main__':
    BASELINE_TIME = 0.5
    vServer = VerifyServer()
    speed = 0
    speed = vServer.time_counter()
    # acc_rate = acc_time = 0.0
    # if speed <= BASELINE_TIME:  # 速度不达标，不进行后续验证
    #     acc_rate, acc_time = vServer.acc_counter()
    print(speed)
    # print(acc_rate)
    # print(acc_time)
    # 结果保存
    # vServer.save(speed, acc_rate, acc_time)