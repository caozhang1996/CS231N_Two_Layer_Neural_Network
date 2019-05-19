#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:03:11 2019

@author: caozhang
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


class TwoLayerNet(object):
    """
    """
    def __init__(self, input_size, hidden_size, output_size, weight_scale=0.0001):
        """
         Inputs:
         - input_size: The dimension D of the input data.
         - hidden_size: The number of neurons H in the hidden layer.
         - output_size: The number of classes C.
        """
        self.params = {}
        self.params['w1'] = weight_scale * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_scale * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    
    def loss(self, X, Y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
    
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - Y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
    
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
    
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        N, D = X.shape
        scores = None
        hidden_out = np.maximum(0, X.dot(w1) + b1)  # relu activation function   
        scores = hidden_out.dot(w2) + b2            # 输出层不需要激活函数
        
        if Y is None:
            return scores
        
        # compute loss
        loss = None
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        probs = exp_scores / sum_exp_scores
        data_loss = np.sum(-np.log(probs[range(N), Y])) / N
        reg_loss = 0.5 * reg * (np.sum(w1 * w1) + np.sum(w2 * w2))
        loss = data_loss + reg_loss
        
        # backward pass: compute gradients
        grads = {}
        dscores = probs.copy()                 # 计算在得分上的梯度
        dscores[np.arange(N), Y] -= 1
        dscores /= N
        
        dw2 = np.dot(hidden_out.T, dscores)
        db2 = np.sum(dscores, axis=0) 
        
        dhidden = np.dot(dscores, w2.T)                # 对隐藏层变量求梯度
        dhidden[hidden_out <= 0] = 0
        
        dw1 = np.dot(X.T, dhidden) 
        db1 = np.sum(hidden_out, axis=0)
        dw2 += reg * w2
        dw1 += reg * w1
        
        grads['w1'] = dw1
        grads['b1'] = db1
        grads['w2'] = dw2 
        grads['b2'] = db2
        return loss, grads
        
      
    def train(self, X, Y, X_val, Y_val, 
              lr_rate=1e-3, reg_strength=1e-5, lr_rate_decay=0.95,
              num_iters=1000, batch_size=200, verbose=False):
        """
         Inputs:
             - X: A numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.
             - y: A numpy array of shape (N,) containing training labels; y[i] = c
             means that X[i] has launtitled2bel 0 <= c < C for C classes.
             - lr_rate: (float) learning rate for optimization.
             - reg: (float) regularization strength.
             - num_iters: (integer) number of steps to take when optimizing
             - batch_size: (integer) number of training examples to use at each step.
             - verbose: (boolean) If true, print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        loss_history = []
        train_acc_history = []
        val_acc_history = []
 
        for it in range(num_iters):
            X_batch = None
            Y_batch = None
            
            index = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[index, :]
            Y_batch = Y[index]
            loss, grads = self.loss(X_batch, Y_batch, reg=reg_strength)
            loss_history.append(loss)
            self.params['w2'] -= lr_rate * grads['w2']
            self.params['b2'] -= lr_rate * grads['b2']
            self.params['w1'] -= lr_rate * grads['w1']
            self.params['b1'] -= lr_rate * grads['b1']    
            
            if verbose == True and it % 50 == 0:
                print ('iteration: %d/%d, loss: %f' % (it, num_iters, loss))
                
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # check accuracy
                train_acc = np.mean(self.predict(X_batch) == Y_batch)
                val_acc = np.mean(self.predict(X_val) == Y_val)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                lr_rate *= lr_rate_decay
            
        return {'loss history': loss_history,
                'train acc history': train_acc_history,
                'val acc history': val_acc_history
                }
            
            
    def predict(self, X):
        """
         Use the trained weights of this two-layer network to predict labels for
         data points. For each data point we predict scores for each of the C
         classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
    
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None
        hidden_out = np.maximum(0, X.dot(self.params['w1']) + self.params['b1'])
        scores = hidden_out.dot(self.params['w2']) + self.params['b2']
        
        y_pred = np.argmax(scores, axis=1)
        return y_pred