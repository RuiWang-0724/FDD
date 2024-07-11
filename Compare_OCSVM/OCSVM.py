import numpy as np
import os
import time
import random
from sklearn import svm
import joblib

random.seed(2333)
when = '{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
log_dir = open(os.path.join('FaultDetection','TE','Compare_OCSVM','results','train_test','{}'.format(when)), "w")

# 对原始数据用OCSVM
train_X_path = os.path.join('FaultDetection','TE','data','pro_3mode_21faults_norm','train_normal_X.npy')
train_X = np.load(train_X_path)
# train_X = train_X[:10]
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1]*train_X.shape[2]))
print('train_X path:{}, shape{}'.format(train_X_path, train_X.shape))

test_normal_X_path = os.path.join('FaultDetection','TE','data','pro_3mode_21faults_norm','test_normal_X.npy')
test_normal_X = np.load(test_normal_X_path)
# test_normal_X = test_normal_X[:10]
test_normal_Y = np.zeros(test_normal_X.shape[0], )
test_fault_X_path = os.path.join('FaultDetection','TE','data','pro_3mode_21faults_norm','test_fault_X.npy')
test_fault_X = np.load(test_fault_X_path)
test_fault_Y_path = os.path.join('FaultDetection','TE','data','pro_3mode_21faults_norm','test_fault_Y.npy')
test_fault_Y = np.load(test_fault_Y_path)
print('test_X path:{}, normal shape{}, fault shape{}'.format(test_normal_X_path, test_normal_X.shape, test_fault_X.shape))
test_X = np.vstack((test_normal_X, test_fault_X))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1]*test_X.shape[2]))
test_Y = np.hstack((test_normal_Y, test_fault_Y))

# OCSVM
clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
clf.fit(train_X)
# Y_pred_train = clf.predict(presentations_train)
test_start_time = time.time()
pred_labels = clf.predict(test_X)  #预测值是1/-1
test_time = time.time() - test_start_time
true_labels = np.zeros(test_Y.shape)
true_labels[test_Y == 0] = 1  # 21类的0类是正常类
true_labels[test_Y != 0] = -1  # 21类的其它类是故障类
print(true_labels, pred_labels)
test_correct = sum(np.equal(pred_labels, true_labels))
test_accuracy = test_correct / true_labels.shape[0]
# 正常样本准确率
Normal_acc = sum(np.equal(pred_labels[true_labels==1], true_labels[true_labels==1])) / true_labels[true_labels==1].shape[0]
FAR = 1 - Normal_acc  # 误报率
# 故障样本准确率
FDR = sum(np.equal(pred_labels[true_labels==-1], true_labels[true_labels==-1])) / true_labels[true_labels==-1].shape[0]

for fault in range(21):  # 各类故障的准确率
    fault += 1
    index = np.argwhere(test_Y == fault)
    accuracy = np.sum(pred_labels[index] == -1) / len(index)
    print('Fault {} accuracy: {}/{}={}'.format(fault, np.sum(pred_labels[index] == -1), len(index), accuracy))
    print('Fault {} accuracy: {}/{}={}'.format(fault, np.sum(pred_labels[index] == -1), len(index), accuracy),file=log_dir)
    log_dir.flush()


print('Test Time: {:.3f}, Accuracy: {:.8f} , Fault Accuracy:{:.8f}, Normal Accuracy:{:.8f}, FAR:{:.8f}'.format(
    test_time, test_accuracy, FDR, Normal_acc, FAR))
print('Test Time: {:.3f}, Accuracy: {:.8f} , Fault Accuracy:{:.8f}, Normal Accuracy:{:.8f}, FAR:{:.8f}'.format(
    test_time, test_accuracy, FDR, Normal_acc, FAR), file=log_dir)
log_dir.flush()

save_model_path = os.path.join('FaultDetection','TE','Compare_OCSVM','results','model','{}'.format(when))
joblib.dump(clf, '{}'.format(save_model_path))
print('save trained OCSVM model, path:{}'.format(save_model_path))
print('save trained OCSVM model, path:{}'.format(save_model_path), file=log_dir)
log_dir.flush()

# # Load the model from the file
# model = joblib.load('model_filename.pkl')
# # Now you can use the loaded model to make predictions
# predictions = model.predict(X_test)


# 结果
# (45710, 6724)
# (20040, 6724)
# [-1 -1 -1 ... -1 -1 -1] [ 1.  1.  1. ... -1. -1. -1.]
# test accuracy 0.7083333333333334, FDR 1.0, FAR 1.0

