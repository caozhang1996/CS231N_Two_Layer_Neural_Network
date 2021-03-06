#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 20:06:14 2019

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from math import sqrt, ceil
import numpy as np


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
     Reshape a 4D tensor of image data to a grid for easy visualization.

     Inputs:
      - Xs: Data of shape (N, H, W, C)
      - ubound(上界): Output grid will have values scaled to the range [0, ubound]
      - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W                 # 这一步必须放在循环内
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                high, low = np.max(img), np.min(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid