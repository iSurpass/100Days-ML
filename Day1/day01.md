@[TOC](机器学习100天之旅 第一天---数据预处理)
# 前言
最近想认真搞一搞机器学习，之前一直一位可望而不可即，其实不是，只要我们踏踏实实地走好每一步，我们一定会学好的。
**让我们一起开始这美妙的机器学习100天之旅吧！**

# 机器学习初步---数据预处理

机器学习离不开大量的数据。万事开头难，在对这些数据进行分析前，
我们先学习一下在 Python 中如何导入数据并对它进行预处理。

---
# 1 导入需要的库
利用 Python 进行数据分析所必须的库有两个。
__NumPy__ 包含了各种**数学计算函数**。
__Pandas__ 用于**导入**和**管理**数据集。

```python
import numpy as np
import pandas as pd
```
---
# 2 导入数据集
数据集通常是 .csv 格式。CSV 以纯文本形式保存表格数据，文件的每一行是一条数据记录。
我们使用 Pandas 的 read_csv 方法读取本地 .csv 文件作为一个数据帧（dataframe）
然后从数据帧中制作自变量和因变量的矩阵和向量。
**原数据集**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190531090722344.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMDEwMzYz,size_16,color_FFFFFF,t_70)
```python
# create 独立变量vector
dataset = pd.read_csv("data.csv")
X = dataset.iloc[ : , : -1].values  # 第一个冒号是所有列（row），第二个是所有行（column）除了最后一个(Purchased)
# create 依赖变量vector
Y = dataset.iloc[ : , 3].values # 只取最后一个column作为依赖变量。
```
---
# 3 处理丢失数据（Missing Data）
 在数据集中可能会出现为空的数据，我们不能删除有空数据的列，这样会对我们机器学习的结果造成影响，在data science中我们可以用NaN代替空值，但是在ML中必须要求数据**numeric**。所以我们可以用mean来代替空值。

我们得到的数据可能由于各种原因存在缺失。为了不降低机器学习模型的性能，
我们可以通过一些方法处理这些数据，比如使用整列数据的平均值或中位数来替换丢失的数据。
```python
'''
Imputer 参数解释：
missing_values：缺失值，可以为整数或 NaN ，默认为 NaN
strategy：替换策略，默认用均值 'mean' 替换，还可以选择中位数 'median' 或众数 'most_frequent'
axis：指定轴数，默认 axis = 0 代表列，axis = 1 代表行
'''
 from sklearn.preprocessing import  Imputer

imputer = Imputer(missing_values="NaN",strategy="mean",axis=0)
训练模型，拟合出作为替换值的均值
imputer = imputer.fit(X[ : , 1:3])
处理需要补全的数据
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```
---
# 4 解析分类数据
分类数据是具有标签值而不是数值的变量，取值范围通常固定（类似枚举值），
不能用于模型的数学计算，需要解析成数字。
为实现这个功能，我们从 sklearn.preprocessing 库中导入 LabelEnconder 类。
在对数据集进行处理时候我们会遇到一些包含同类别的数据（如图二中的country）。
这样的数据是非numerical的数据，所以我们可以用数字来代替，比如不同的国家我们可以用1,2,3区分不同国家，但是这样会出现一个比较严重的问题。
就是国家之间的地位是相同的，但是数字有顺序大小之分。
所以我们用另一种方法，就是将不同的类别（如不同国家）另外分为一个列，属于这个国家的设置为1，不属于的设置为0。
```python
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
```
---
# 5 分割数据集
我们需要把数据集拆分成用来训练模型的训练集和用来验证模型的测试集。两者的比例一般是 80：20。

```python
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= 0.2, random_state= 0)
```
---

# 6 特征缩放 （feature scaling）
这是对数据处理的一项很重要的步骤，在机器学习中，由于每个变量的范围不同，
如果两个变量之间差距太大，会导致距离对结果产生影响。
所以我们要对数据进行一定的标准化改变。最简单的方式是将数据缩放至[0.1]或者[-1,1]之间.

```python
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
训练后的模型数据
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190531090641511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMDEwMzYz,size_16,color_FFFFFF,t_70)

