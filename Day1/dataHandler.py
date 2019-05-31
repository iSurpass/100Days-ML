'''
------------ 100-Days-ML -------------
************   01 Day    *************
今日课题: 机器学习初步~数据预处理

机器学习离不开大量的数据。在对这些数据进行分析前，
我们先学习一下在 Python 中如何导入数据并对它进行预处理。
***********************************************
-----------------------------------------------
'''
# -*- coding: utf-8 -*-
# @File  : dataHandler.py
# @Author: Juniors
# @Date  : 2019/5/30

import numpy as np
import pandas as pd


# create 独立变量vector
dataset = pd.read_csv("data.csv")
X = dataset.iloc[ : , : -1].values  # 第一个冒号是所有列（row），第二个是所有行（column）除了最后一个(Purchased)
# create 依赖变量vector
Y = dataset.iloc[ : , 3].values # 只取最后一个column作为依赖变量。

from sklearn.preprocessing import  Imputer

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
# 训练模型，拟合出作为替换值的均值
imputer = imputer.fit(X[ : , 1:3])
# 处理需要补全的数据
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import  OneHotEncoder

labelencoder_X = LabelEncoder()
# 对 X 中的标签数据编码
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
# 使用 onehotencoder 对经过标签编码的第0行数据进行独热编码
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# 对 Y 中的标签数据编码
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, random_state= 0)


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print(dataset)
print("--------------------")
print(X_train)


