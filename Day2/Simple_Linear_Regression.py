'''
------------ 100-Days-ML -------------
************   02 Day    *************
今日课题: 机器学习初步~简单线性回归

回归分析是一种预测性的建模技术，它研究的是因变量（目标）和自变量（预测器）之间的关系。
这种技术通常用于预测分析，时间序列模型以及发现变量之间的因果关系。
例如，司机的鲁莽驾驶与道路交通事故数量之间的关系，最好的研究方法就是回归。
回归分析是建模和分析数据的重要工具。在这里，我们使用曲线/线来拟合这些数据点，
在这种方式下，从曲线或线到数据点的距离差异最小。
***********************************************
-----------------------------------------------
'''
# -*- coding: utf-8 -*-
# @File  : Simple_Linear_Regression.py
# @Author: Juniors
# @Date  : 2019/5/31

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dataset = pd.read_csv("studentscore.csv")
X = dataset.iloc[ : , : 1].values
Y = dataset.iloc[ : , 1].values
print("X:",X)
print("Y",Y)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test,  = train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

plt.title("Hours Vs Score (Training set)")
plt.xlabel("Studying Hours")
plt.ylabel("Student's score")
plt.scatter(X_train,Y_train,color = "red")
plt.plot(X_train,regressor.predict(X_train),'bo-')
plt.show()

plt.title("Hours Vs Score (Testing set)")
plt.xlabel("Studying Hours")
plt.ylabel("Student's score")
plt.scatter(X_test,Y_test,color= "red")
plt.plot(X_test,Y_pred,'bo-')
plt.show()




