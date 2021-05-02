import random
import sys

from PIL import Image
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c


def error(params, x, y):
    return func(params, x) - y


def slovePara():
    p0 = [10, 10, 10]

    Para = leastsq(error, p0, args=(X, Y))
    return Para


def solution():
    Para = slovePara()
    a, b, c = Para[0]
    print("a=", a, " b=", b, " c=", c)
    print("cost:" + str(Para[1]))
    print("求解的曲线是:")
    print("y=" + str(round(a, 2)) + "x*x+" + str(round(b, 2)) + "x+" + str(c))
    plt.figure(figsize=(8, 6))
    plt.scatter(X, Y, color="green", label="sample data", linewidth=2)
    plt.legend()  # 绘制图例
    # plt.show()

    return a, b, c


def plt_best():
    x = np.linspace(0, 256, 500)  ##在0-256直接画500个连续点
    y = cur_a * x * x + cur_b * x + cur_c  # 当前拟合的y值
    plt.title("fin")
    plt.scatter(cur_X, cur_Y, color="green", label="sample data", linewidth=2)
    plt.plot(x, y, color="red", label="solution line", linewidth=2)
    plt.legend()  # 绘制图例
    plt.show()


no = 1
cnt = 100
for no in range(3,4):
    train = Image.open("./train/%d.png" % no)
    label = Image.open("./label/%d.png" % no)
    train_array = np.asarray(train)
    black = []
    for i in range(256):
        for j in range(256):
            if train_array[i][j] < 100:
                black.append((i, j))
    # print(black)
    black_set = set(black)
    for tup in black_set.copy():
        if tup[0] < 40:
            black_set.remove(tup)
    black = list(black_set)
    unique_x_dict = {}
    for tup in black:
        if tup[0] in unique_x_dict.keys():
            unique_x_dict[tup[0]] = (unique_x_dict[tup[0]] + tup[1]) / 2
        else:
            unique_x_dict[tup[0]] = tup[1]
    cur_a = sys.maxsize
    cur_b = sys.maxsize
    cur_c = sys.maxsize
    cur_loss = sys.maxsize
    cur_X = []
    cur_Y = []
    for it in range(10):
        random.seed
        res = random.sample(range(0, len(black)), cnt)
        # print(res)
        # print(black[res[0]])
        X = np.zeros(cnt)
        Y = np.zeros(cnt)
        cnt2 = 0
        for r in res:
            curtup = black[r]
            # print(curtup)
            X[cnt2] = curtup[1]
            Y[cnt2] = 256 - curtup[0]
            cnt2 = cnt2 + 1
        a, b, c = solution()
        # 画拟合线
        x = np.linspace(0, 256, 256)  ##在0-256直接画500个连续点
        y = a * x * x + b * x + c  # 当前拟合的y值
        loss = 0
        x_index = 0
        for this_y in y:
            if x_index in unique_x_dict.keys():
                loss = loss + abs(this_y - unique_x_dict[x_index])
                loss_added = abs(this_y - unique_x_dict[x_index])
                print(loss_added)
            x_index = x_index + 1
        print("loss: ", loss)
        if loss < cur_loss:
            cur_loss = loss
            cur_a = a
            cur_b = b
            cur_c = c
            cur_Y = Y
            cur_X = X
    print("min_loss: ", cur_loss)
    print("fin_a: ", cur_a)
    print("fin_b: ", cur_b)
    print("fin_c: ", cur_c)
    plt_best()

