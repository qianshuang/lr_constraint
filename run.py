# -*- coding: utf-8 -*-

from sklearn import metrics, linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

breast_cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.1)
model = linear_model.LogisticRegression(penalty='l1')  # ovr


# 训练
def train():
  model.fit(x_train, y_train)


# 测试
def test():
  test_predict = model.predict(x_test)  # 返回预测类别
  test_predict_proba = model.predict_proba(x_test)  # 返回属于各个类别的概率
  test_predict_true_proba = test_predict_proba[:, 1]  # true probability

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

  # AUC
  test_target = list(map(float, y_test))
  fpr, tpr, thresholds = metrics.roc_curve(test_target, test_predict_true_proba)
  print("\nAUC...")
  print(metrics.auc(fpr, tpr))


# train()
# test()
print(np.mat(np.ones(5)))
