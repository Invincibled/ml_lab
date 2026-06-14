# PCA
# 什么样的特征，是比较主要的呢，或是适合的，分布广，方差大的；
from  sklearn.decomposition import PCA

import numpy as np

import matplotlib.pyplot as plt

# X = np.random.randn(1000, 3)
# print(X.shape)
#
# # 使用PCA 进行降维，将3维降维2
#
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# print(X_pca.shape)
#
# # 可视化转，3维转2维
#
# figure = plt.figure(figsize=(12, 4))
#
# ax1 = figure.add_subplot(121, projection='3d')
# ax1.step(X[:, 0], X[:, 1], X[:, 2], color='g')
# ax1.set_title('Before PCA(3D)')
# ax1.set_xlabel('Feature1')
# ax1.set_ylabel('Feature2')
# ax1.set_zlabel('Feature3')
#
# # plt.show()
#
# ax2 = figure.add_subplot(122)
# ax2.step(X_pca[:, 0], X_pca[:, 1], color='g')
# ax2.set_title('After PCA(2D)')
# ax2.set_xlabel('Principal Component 1')
# ax2.set_ylabel('Principal Component 2')
#
# plt.show()

# 手动构建线性相关的3组特征数据
n = 1000
pc1 = np.random.normal(0, 1, n)

pc2 = np.random.normal(0, 0.2 , n)

# 定义不重要的第三组成分

noise = np.random.normal(0, 0.05, n)

# 构建三个特征
X1 = np.vstack((pc1+pc2, pc1-pc2, pc2+noise)).T
print(X1)

pca = PCA(n_components=2)
X1_pca = pca.fit_transform(X1)
print(X1_pca.shape)


figure = plt.figure(figsize=(12, 4))

ax1 = figure.add_subplot(121, projection='3d')
ax1.step(X1[:, 0], X1[:, 1], X1[:, 2], color='g')
ax1.set_title('Before PCA(3D)')
ax1.set_xlabel('Feature1')
ax1.set_ylabel('Feature2')
ax1.set_zlabel('Feature3')

# plt.show()

ax2 = figure.add_subplot(122)
ax2.step(X1_pca[:, 0], X1_pca[:, 1], color='g')
ax2.set_title('After PCA(2D)')
ax2.set_xlabel('Principal Component 1')
ax2.set_ylabel('Principal Component 2')

plt.show()









