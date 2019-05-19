#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:48:23 2019

@author: caozhang
"""


import numpy as np

a = np.array([[1, 2, 3],
              [5, 6, 7],
              [3, 2, 1]])
b = np.array([[0, 1, 1],
              [4, 5, 6],
              [2, 1, 0]])
print (b - a)
print (np.abs(b - a))
print (np.abs(-0.923))
print (np.sum(np.abs(b - a)))
print (np.sum(b - a))


x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14,15],
              [16, 17, 18],
              [19, 20, 21],
              [22, 23, 24],
              [25, 26, 27],
              [28, 29, 30],
              [31, 32, 33],
              [34, 35, 36]])

print (x)
print (x.reshape(2, 2, 3, -1))
print (x.reshape(2, 2, 3, -1).transpose(3, 0, 1, 2))
