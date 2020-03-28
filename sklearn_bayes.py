# ! /user/bin/env python
# # -*- coding: utf-8 -*-

"""
sklearn 使用朴素贝叶斯分类器
"""

#### 1、高斯朴素贝叶斯算法
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as CM

test_size=0.3
digits=load_digits()
x,y=digits.data,digits.target

train_X,test_X,train_Y,test_Y=train_test_split(x,y,test_size=test_size)
print(train_X[:10])
print(train_Y[:10])
gnb=GaussianNB().fit(train_X,train_Y)
acc_score=gnb.score(test_X,test_Y)
print(acc_score)
pred_Y=gnb.predict(test_X)
# print(pred_Y)
prob=gnb.predict_proba(test_X)
# print(prob)
# print(prob[1,:].sum())

print(CM(test_Y,pred_Y))
