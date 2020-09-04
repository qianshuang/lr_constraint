# -*- coding: utf-8 -*-

from sklearn import metrics, linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


def sigmoid(inX):
  return 1.0 / (1 + np.exp(-inX))  # 使用sigmoid函数


def logistic_regression(x_train, y_train, k):  # k是迭代次数
  m, n = np.shape(x_train)  # 样本数据有m个，加上偏置项有特征值n个
  X = x_train.T
  Y = y_train
  theta = np.mat(np.ones(n))  # 初始化权重矩阵
  for i in range(k):
    theta = theta - 0.2 / m * (sigmoid(theta * X) - Y) * X.T  # 梯度更新公式
    # 权重保序约束
    theta = np.sort(theta, axis=-1)
  return theta


def classify(x_test, theta):
  prob = sigmoid(theta * x_test.T)
  return prob


# load data
breast_cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.1)

# train
theta = logistic_regression(x_train, y_train, 5000)

# test
test_predict_true_proba = classify(x_test, theta)  # true probability
test_predict = []
for i in test_predict_true_proba.A[0]:
  if i >= 0.5:
    test_predict.append(1)
  else:
    test_predict.append(0)

# accuracy
true_false = (test_predict == y_test)
accuracy = np.count_nonzero(true_false) / float(len(y_test))
print()
print("accuracy is %f" % accuracy)

# precision    recall  f1-score
print()
print(metrics.classification_report(y_test, test_predict, target_names=['0', '1']))

# 混淆矩阵
print("Confusion Matrix...")
print(metrics.confusion_matrix(y_test, test_predict))
