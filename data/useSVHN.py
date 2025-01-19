#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 09:55:36 2017
@author: cheers
"""
 
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
 
image_size = 32
num_labels = 10
 
def display_data():
    train = sio.loadmat('train_32x32.mat')
    data=train['X']
    label=train['y']
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.title(label[i][0])
        plt.imshow(data[...,i])
        plt.axis('off')
    plt.show()
 
def load_data(one_hot = False):
    
    train = sio.loadmat('train_32x32.mat')
    test = sio.loadmat('test_32x32.mat')
    
    train_data=train['X']
    train_label=train['y']
    test_data=test['X']
    test_label=test['y']
    
    
    train_data = np.swapaxes(train_data, 0, 3)
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)
    test_data = np.swapaxes(test_data, 0, 3)
    test_data = np.swapaxes(test_data, 2, 3)
    test_data = np.swapaxes(test_data, 1, 2)
    
    test_data = test_data / 255.
    train_data =train_data / 255.
    
    for i in range(train_label.shape[0]):
         if train_label[i][0] == 10:
             train_label[i][0] = 0
                        
    for i in range(test_label.shape[0]):
         if test_label[i][0] == 10:
             test_label[i][0] = 0
 
    if one_hot:
        train_label = (np.arange(num_labels) == train_label[:,]).astype(np.float32)
        test_label = (np.arange(num_labels) == test_label[:,]).astype(np.float32)
 
    return train_data,train_label, test_data,test_label
 
if __name__ == '__main__':
    load_data(one_hot = True)
    display_data()