import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os
os.environ['DISPLAY'] = 'localhost:10.0'

res = [re.compile('.*Epoch\[(\d+)\] .*Train-acc.*=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Validation-acc.*=([.\d]+)')]


def plot_acc(log_name, color="r"):

    train_name = log_name.replace(".log", " train")
    val_name = log_name.replace(".log", " val")

    data = {}
    with open(log_name) as f:
        lines = f.readlines()
    for l in lines:
        i = 0
        for r in res:
            m = r.match(l)
            if m is not None:  # i=0, match train acc
                break
            i += 1  # i=1, match validation acc
        if m is None:
            continue
        assert len(m.groups()) == 2
        epoch = int(m.groups()[0])
        val = float(m.groups()[1])
        if epoch not in data:
            data[epoch] = [0] * len(res) * 2
        data[epoch][i*2] += val  # data[epoch], val:number
        data[epoch][i*2+1] += 1

    train_acc = []
    val_acc = []
    for k, v in data.items():
        if v[1]:
            train_acc.append(v[0]/(v[1]))
        if v[2]:
            val_acc.append(v[2]/(v[3]))

    x_train = np.arange(len(train_acc))
    x_val = np.arange(len(val_acc))
    plt.plot(x_train, train_acc, '-', linestyle='--', color=color, linewidth=2, label=train_name)
    plt.plot(x_val, val_acc, '-', linestyle='-', color=color, linewidth=2, label=val_name)
    plt.legend(loc="best")
    plt.xticks(np.arange(0, 40, 1))
    plt.yticks(np.arange(0.5, 1.0, 0.01))
    plt.xlim([0, 40])
    plt.ylim([0.5, 1.0])

def main():
    plt.figure(figsize=(14, 8))
    plt.xlabel("epoch")
    plt.ylabel("Top-1 error")
    color = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    log_files = [i for i in args.logs.split(',')]
    color = color[:len(log_files)]
    for c in range(len(log_files)):
        plot_acc(log_files[c], color[c])
    plt.grid(True)
    plt.show()
    # plt.savefig(args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', type=str, default="resnet-101-aic-0005-336-119-acapfixall.log",)

    # parser.add_argument('--logs', type=str, default="resnet-101-aic-0005-336-119-acapfixall.log",)
    # parser.add_argument('--logs', type=str, default="resnet-152-aic-001-448-acap-meituaiconelabel-1199.log",)
    parser.add_argument('--out', type=str, default="training-curve.png")
    args = parser.parse_args()
    main()
