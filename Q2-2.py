# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:22:18 2022

@author: Johnny
"""
import numpy as np
import pandas as pd
from numpy.random import randn
import math
import cv2
import matplotlib.pyplot as plt


def load_img_data_by_txt(img_load_list):
    tmp_hist_df = []
    for index, tmp_img_path in enumerate(img_load_list):
        temp_img = cv2.imread(tmp_img_path)
        temp_img = cv2.resize(temp_img, (100,100))
        colors = ('b', 'g', 'r')
        
        tmp_hist_array = np.array([])
        for i, col in enumerate(colors):
            hist = cv2.calcHist([temp_img], [i], None, [256], [0, 256])
            hist = hist.flatten()
            tmp_hist_array = np.append(tmp_hist_array, hist)
        if index == 0:
            tmp_hist_df = pd.DataFrame(tmp_hist_array).T
        else:
            tmp_hist_df = tmp_hist_df.append(pd.DataFrame(tmp_hist_array).T)
    tmp_train_x = tmp_hist_df.reset_index(drop = True)
    return tmp_train_x

def model_fit(x, w1, w2):
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    return y_pred 

def softmax(y_pred):
    for row in range(len(y_pred)):
        y_pred.iloc[row, :] = np.exp(y_pred.iloc[row, :]) / np.sum(np.exp(y_pred.iloc[row, :]))
    return y_pred

def cross_entropy(y_pred, y):
    loss = 0
    for index, label in enumerate(y):
        loss += np.log2(y_pred.iloc[index, label]+0.0001)
    return (-1)*(loss)

def gradient(y_pred, y):
    grad_y_pred = y_pred
    for index, label in enumerate(y):
        grad_y_pred.iloc[index, label] -= 1
    return grad_y_pred

def get_accuracy(data_list, batch_size, w1, w2): 
    correct = 0
    total = len(data_list)
    for j in range(math.ceil(len(data_list) / batch_size)):
        start_point = 0
        end_point = 1*batch_size
        
        if end_point > len(data_list):
            end_point = len(data_list)
        
        tmp_load_list = list(data_list.loc[start_point : end_point - 1, 'path'])
        x = load_img_data_by_txt(tmp_load_list)
        y = data_list.loc[start_point : end_point - 1, 'label']
    
        y_pred = softmax(model_fit(x, w1, w2))
    
        for index, label in enumerate(y):
            if np.argmax(y_pred.iloc[index, :]) == label:
                correct += 1
    
    accuracy = correct / total
    return accuracy
    
## Settings
batch_size = 32
epochs = 1
learning_rate = [0.002, 0.002, 0.002]
N, D_in, H, D_out = batch_size, 768, 256, 50
w1, w2 = randn(D_in, H), randn(H, D_out)
train_acc, val_acc, test_acc = [], [], []

## Import Path
train_txt = pd.read_csv('train.txt', header = None, names = ['path', 'label'], sep = ' ')
val_txt = pd.read_csv('val.txt', header = None, names = ['path', 'label'], sep = ' ')
test_txt = pd.read_csv('test.txt', header = None, names = ['path', 'label'], sep = ' ') 
    
## train model
loss_info = []

train_txt_shuffled = train_txt.sample(frac = 1).reset_index(drop = True)
# Load Data by batch
for i in range(math.ceil(len(train_txt_shuffled) / batch_size)):
    start_point = 0
    end_point = 1*batch_size
        
    if end_point > len(train_txt_shuffled):
        end_point = len(train_txt_shuffled)
        
    tmp_load_list = list(train_txt_shuffled.loc[start_point : end_point - 1, 'path'])
    train_x = load_img_data_by_txt(tmp_load_list) # train_x = load_img_data_by_txt(train_txt['path'])
    train_y = train_txt_shuffled.loc[start_point : end_point - 1, 'label']
    
    # predict
    h = 1 / (1 + np.exp(-train_x.dot(w1)))
    y_pred = h.dot(w2)
        
    # softmax
    y_pred = softmax(y_pred)
        
    # Cross-entropy loss
    loss = cross_entropy(y_pred, train_y)
    print('epoch:', epochs, 'iter:', i,',loss:', loss)
    loss_info.append(loss)
        
    # get accuracy
    if i % 50 == 0:
        train_acc.append(get_accuracy(train_txt, batch_size, w1, w2))
        val_acc.append(get_accuracy(val_txt, batch_size, w1, w2))
        test_acc.append(get_accuracy(test_txt, batch_size, w1, w2))
            
    # Update gradients
    grad_y_pred = gradient(y_pred, train_y)
        
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = train_x.T.dot(grad_h * h * (1 - h))
        
    # Update weights
    w1 -= learning_rate[0] * grad_w1
    w2 -= learning_rate[0] * grad_w2  

plt.plot(loss_info)
plt.title('Triaining Curve(batch size = 32, lr = 0.001)')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.savefig('Training.png')
plt.close()

plt.plot(train_acc, label = 'train')
plt.plot(val_acc, label = 'val')
plt.plot(test_acc, label = 'test')
plt.title('Accuracy')
plt.legend()
plt.savefig('Accuracy.png')
plt.close()