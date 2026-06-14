
# 斯皮尔曼相关系数法
# 取值范围 【-1,1]
# 1. 完全相关
# -1 完全负相关
# 0 无相关性
# 斯皮尔曼相关法，是根据判定的等级，然后看等级差

import pandas as pd
import numpy as np


X = [[5], [8], [10], [12], [15], [3], [7], [9], [14],[6]]
y = [55,65, 70,75,85,50,60,72,80,58]
X = pd.DataFrame(X)
y = pd.Series(y)
print(X.shape)
print(y.shape) 

print(X.corrwith(y, method='spearman'))


# df = pd.read_csv('./data/data_advertising.csv')
# df.drop(df.columns[0], axis=1, inplace=True)
# df.dropna(inplace=True)
# # 提取特征和标签(目标值)
# X = df.drop("sales", axis=1)
# y = df["sales"]
#
# # 计算相关系数
# r = X.corrwith(y, method='spearman')
# print(r)
# print(df.corr(method='spearman'))
#
# # 将相关系数矩阵，画成热力图
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# sns.heatmap(df.corr(method='spearman'), annot=True)
# plt.show()

