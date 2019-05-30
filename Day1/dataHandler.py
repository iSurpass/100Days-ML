# -*- coding: utf-8 -*-
# @File  : dataHandler.py
# @Author: Juniors
# @Date  : 2019/5/30
'''
------------ 100-Days-ML -------------
************   01 Day    *************
今日课题: 机器学习初步~数据预处理

'''
import numpy as np
import pandas as pd

dataset = pd.read_csv("data.csv")
X = dataset.iloc[ : , : -1].values
Y = dataset.iloc[ : , 3].values

from sklearn.preprocessing import  Imputer

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import  OneHotEncoder

labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

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


