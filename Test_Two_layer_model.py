#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:45:43 2019

@author: caozhang
创建一个小的网络和测试数据来检查TwoLayerNet代码是否能正确实现
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from classifier.neural_net import TwoLayerNet
from data_utils import load_CIFAR10


input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5


def init_toy_model():
    # 如果在seed()中传入的数字相同，那么接下来使用random()或者rand()方法所生成的随机数序列都是相同的
    #（仅限使用一次random()或者rand()方法，第二次以及更多次仍然是随机的数字）
    np.random.seed(0)         # 随机数种子
    return TwoLayerNet(input_size, hidden_size, num_classes, weight_scale=1e-1)


def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    Y = np.array([0, 1, 2, 2, 1])
    return X, Y


Net = init_toy_model()
X, Y = init_toy_data()

# 计算得分
scores = Net.loss(X)
print ('Your socres:')
print (scores)
print ('correct socres:')
correct_scores = np.array([[-0.81233741, -1.27654624, -0.70335995],
                           [-0.17129677, -1.18803311, -0.47310444],
                           [-0.51590475, -1.01354314, -0.8504215 ],
                           [-0.15419291, -0.48629638, -0.52901952],
                           [-0.00618733, -0.12435261, -0.15226949]])
print (correct_scores)
print ('Difference between your scores and correct scores:')
print (np.sum(np.abs(scores - correct_scores)))

# 计算损失
loss, _ = Net.loss(X, Y, reg=0.1)
correct_loss = 1.30378789133
print ('Your loss:')
print (loss)
print ('Correct loss:')
print (correct_loss)
print ('Difference between your loss and correct loss:')
print (np.abs(loss - correct_loss))

# 开始训练
net = init_toy_model()
stats = net.train(X, Y, X, Y, lr_rate=1e-1, reg_strength=1e-5, num_iters=100, verbose=False)
print ('Final training loss: ', stats['loss history'][-1])

# 画出迭代过程中的损失值变化图像
plt.plot(stats['loss history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training loss history')
plt.show()
