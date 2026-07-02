import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '2'

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sqlalchemy import label
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 1. 生成数据
X, y = make_blobs(n_samples=300, centers=3, cluster_std=2.0)


# 画出散点图
fig, ax = plt.subplots(2, figsize=(10, 8))
ax[0].scatter(X[:, 0], X[:, 1], c="gray", s=50, label="原始数据")
ax[0].set_title("原始数据")
ax[0].legend()


# 2. KMeans聚类
kmeans = KMeans(n_clusters=3, random_state=0)

kmeans.fit(X)
# 获取结果
centers = kmeans.cluster_centers_

y_pred = kmeans.predict(X)

ax[1].scatter(X[:, 0], X[:, 1], c=y_pred, s=50, label="KMeans聚类")
ax[1].scatter(centers[:, 0], centers[:, 1], c="red", s=200, label="质心")
ax[1].set_title("KMeans聚类")
ax[1].legend()

plt.show()
