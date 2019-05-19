#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:44:48 2019

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from  classifier.neural_net import TwoLayerNet
import data_utils
from vis_utils import visualize_grid


def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=1000, num_dev=500):
    """
    """
    root_dir = '../dataset/cifar-10-batches-py'
    X_train , Y_train, X_test, Y_test = data_utils.load_CIFAR10(root_dir)
    
    mask = list(range(num_training, num_training + num_val))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]
    
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]
    
    # reshape the images
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # 使X_train, X_dev, X_val, 中的数据零均值化
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_dev -= mean_image
    X_test -= mean_image
    return X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test


def training():
    """
    """
    X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test = get_CIFAR10_data()
    input_size = 32 * 32 * 3
    hidden_size = 280
    num_classes = 10
    Net = TwoLayerNet(input_size, hidden_size, num_classes)
    
    result = Net.train(X_train, Y_train, X_val, Y_val, lr_rate=1.6e-3, lr_rate_decay=0.9,
                       reg_strength=0.5, num_iters=3000, batch_size=400, verbose=True)
    
    y_val_predict = Net.predict(X_val)
    val_acc = np.mean(y_val_predict == Y_val)
    print ('Validation accuracy: %f' % val_acc)
    y_test_predict = Net.predict(X_test)
    test_acc = np.mean(y_test_predict == Y_test)
    print ('Test accuracy : %f' % test_acc)
    
    # 绘制损失值
    plt.subplot(211)
    plt.plot(result['loss history'])
    plt.xlabel('Iteration'), plt.ylabel('Loss value')
    plt.title('Loss history')
    
    plt.subplot(212)
    plt.plot(result['train acc history'], label='train')
    plt.plot(result['val acc history'], label='val')
    plt.xlabel('Epoch'), plt.ylabel('Accuracy value')
    plt.title('Accuracy history')
    plt.show()
    
    # 将第一层中w1可视化，有32 * 32 * 3个隐藏层神经元，应有32 * 32 * 3张图片
    w1 = Net.params['w1']
    w1 = w1.reshape((32, 32, 3, -1)).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(w1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def hyperparameter_tuning():
    """
    """
    X_train, Y_train, X_val, Y_val, X_dev, Y_dev, X_test, Y_test = get_CIFAR10_data()
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    
    best_net = None
    best_val = -1
    learning_rates = [0.75e-4, 1.5e-4, 1.25e-4, 1.75e-4, 2e-4]
    reg_strengths = [0.25, 0.3, 0.35, 0.4, 0.45]
    results = {}
    iters = 2000
    
    for lr in learning_rates:
        for reg in reg_strengths:
            net = TwoLayerNet(input_size, hidden_size, num_classes)
            stats = net.train(X_train, Y_train, X_val, Y_val,
                              num_iters=iters, batch_size=200,
                              lr_rate=lr, reg_strength=reg,
                              lr_rate_decay=0.95)
            y_train_pred = net.predict(X_train)
            train_acc = np.mean(y_train_pred == Y_train)
            y_val_pred = net.predict(X_val)
            val_acc = np.mean(y_val_pred == Y_val)
            results[(lr, reg)] = (train_acc, val_acc)
            
            if best_val < val_acc:
                best_val = val_acc
                best_net = net 
    
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print ('lr', lr, 'reg', reg, 'train accuracy', train_accuracy, 'val accuracy', val_accuracy)
        
    print ('best validation accuracy: ', best_val)
