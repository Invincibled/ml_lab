from sklearn.feature_selection import VarianceThreshold
import numpy as np
# 低方差过滤法

# 删除方差低于0.01的特征
# var_thresh = VarianceThreshold(threshold=0.01)
# x_filter = var_thresh.fit_transform(X)


# 构建特征
a = np.random.rand(100)
# print(a)
# 打印方差
# print(np.var(a))

b = np.random.rand(100) * 0.1
# print(b)
# print(np.var(b))

c = np.random.normal(5,0.1 , size=100)
# print(c)
# print(np.var(c))


# X = np.stack((a, c)).T
# print(X)
# print(X.shape)
#
# var_thresh = VarianceThreshold(threshold=0.01)
# x_filtered = var_thresh.fit_transform(X)
# print(x_filtered)
# print(x_filtered.shape)


# 皮尔逊相关系数，取值范围 [-1,1]
# 1 正相关
# -1 负相关
# 0 无关
import pandas as pd

df = pd.read_csv('../data/data_advertising.csv')
df.drop(df.columns[0], axis=1, inplace=True)
df.dropna(inplace=True)
# 提取特征和标签(目标值)
X = df.drop("sales", axis=1)
y = df["sales"]

# 计算相关系数
r = X.corrwith(y, method='pearson')
print(r)
print(df.corr(method='pearson'))

# 将相关系数矩阵，画成热力图
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(method='pearson'), annot=True)
plt.show()

