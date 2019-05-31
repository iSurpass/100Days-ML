'''
------------ 100-Days-ML -------------
************   03 Day    *************
今日课题: 机器学习初步~多元线性回归

多元线性回归（Multiple Linear Regression）
尝试通过已知数据找到一个线性方程来描述两个及以上的特征（自变量）与
输出（因变量）之间的关系，并用这个线性方程来预测结果。
***********************************************
-----------------------------------------------
'''
# -*- coding: utf-8 -*-
# @File  : Multiple_Linear_Regression.py
# @Author: Juniors
# @Date  : 2019/5/31

import numpy as np
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4 ].values

# print(X[:10])
# print(Y)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[: , 3])
# print("labelencoder:")
# print(X[ :10])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
# print("onehotencoder:")
# print(X[:10])

#躲避虚拟变量陷阱
X1 = X[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

reggressor = LinearRegression()
reggressor.fit(X_train,Y_train)
reggressor1 = LinearRegression()
reggressor1.fit(X1_train,Y1_train)

y_predict = reggressor.predict(X_test)
y1_predict = reggressor1.predict(X1_test)

# print(y_predict)
# print(y1_predict)








