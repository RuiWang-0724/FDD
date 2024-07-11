from BaseSVDD import BaseSVDD
import numpy as np
import torch
import time
import os

from function import MyDataset, t_SNE
from sklearn import svm

# 加载数据并均衡
train_X = np.load('/home/priceless/WR/FaultDetection/TE/data/processed_6domain_normalization/train_normal_X.npy')
np.random.shuffle(train_X)
train_X = np.reshape(train_X[0:1000], (train_X[0:1000].shape[0], train_X[0:1000].shape[1]*train_X[0:1000].shape[2]))# 选择作为标准的数据数量

test_normal_X = np.load('/home/priceless/WR/FaultDetection/TE/data/processed_6domain_normalization/test_normal_X.npy')
test_normal_Y = np.zeros(test_normal_X.shape[0],)
print(test_normal_Y.shape, test_normal_X.shape)
test_fault_X = np.load('/home/priceless/WR/FaultDetection/TE/data/processed_6domain_normalization/test_fault_X.npy')
test_fault_Y = np.load('/home/priceless/WR/FaultDetection/TE/data/processed_6domain_normalization/test_fault_Y.npy')
label = []
fault = []
for cls in range(28):
    cls += 1
    index = (test_fault_Y == cls)
    fault_X = test_fault_X[index]
    np.random.shuffle(fault_X)
    fault_X = fault_X[0:round(test_normal_X.shape[0]/28)]
    # print('类别', cls, '数据量', fault_X.shape)
    y = cls * np.ones((fault_X.shape[0],))
    label.append(y)
    fault.append(fault_X)
test_fault_X_equal = np.vstack(fault)
test_fault_Y_equal = np.hstack(label)
print(test_fault_Y_equal.shape, test_fault_X_equal.shape)
test_X = np.vstack((test_normal_X, test_fault_X_equal))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1]*test_X.shape[2]))
test_Y = np.hstack((test_normal_Y, test_fault_Y_equal))

# SVDD
svdd = BaseSVDD(C=0.001, gamma=0.5, kernel='rbf', display='on')
svdd.fit(train_X)
Y_pred_test = svdd.predict(test_X)

average_accuracy = (np.sum(Y_pred_test[(test_Y != 0)] == -1) + np.sum(Y_pred_test[(test_Y == 0)] == 1)) / len(test_Y)
FDR_index = np.argwhere(test_Y != 0)
FDR = np.sum(Y_pred_test[FDR_index] == -1) / len(FDR_index)  # Fault Detection Rate
FAR_index = np.argwhere(test_Y == 0)
FAR = np.sum(Y_pred_test[FAR_index] == -1) / len(FAR_index)  # False Alarm Rate
print(Y_pred_test, test_Y)
print('test accuracy {}, FDR {}, FAR {}'.format(average_accuracy, FDR, FAR))

for cls in range(28): # 各类故障的准确率
    cls += 1
    index = np.argwhere(test_Y == cls)
    accuracy = np.sum(Y_pred_test[index] == -1)/len(index)
    print('Fault {} accuracy: {}/{}={}'.format(cls, np.sum(Y_pred_test[index] == -1), len(index), accuracy))