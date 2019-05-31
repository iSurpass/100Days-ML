@[TOC](机器学习100天之旅 第一天---数据预处理)
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