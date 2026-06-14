
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2,1],[3,1],[1,4],[2,6]])  # 数据准备
y = np.array([0,1,0,1]) # 分类标签
knn = KNeighborsClassifier(n_neighbors=2, weights='distance')

knn.fit(X, y)

x = np.array([[4,9]])
x_class = knn.predict(x)
print(x_class)

# 画图
fig, ax = plt.subplots()
ax.axis('equal')

X1 = X[y==0]
X2 = X[y==1]
colors = ["C0", "C1"]
plt.scatter(X1[:,0], X1[:,1], color=colors[0])
plt.scatter(X2[:,0], X2[:,1], color=colors[1])
x_color = colors[0] if x_class == 0 else colors[1]
plt.scatter(x[0][0], x[0][1], color=x_color)
plt.show()