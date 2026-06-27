import numpy as np


# 定义损失函数
def J(beta):
    return np.sum((X @ beta - y) ** 2) / n

# 定义梯度函数
def gradient(beta):
    return  X.T @ (X @ beta - y) / n * 2

# 1. 定义数据
X = np.array([[5], [8], [10], [12], [15], [3], [7],[9], [14], [6]])
# 自变量,每周学习时长
y = np.array([[55], [65], [70], [75], [85], [50], [60],[72],[80], [58]]) #因变量,数学考试成绩
# 矩阵行数
n = X.shape[0]

# 2. 数据处理，X增加一列 1
X = np.hstack((np.ones((n, 1)), X))


# 3. 初始化参数及超参数
alpha = 0.01
iter = 10000
beta = np.array([[1],[1]])

# 定义列表，保存参数变化轨迹.
beta0 =[]
beta1 =[]
# 重复迭代
# for i in range(iter):
#     # 计算梯度
#     grad  = gradient(beta)
#     # 更新参数
#     beta = beta - alpha * grad
#     # 每迭代10次打印一次当前参数和损失值
#     if i % 10 == 0:
#         print(f"iteration {i}, beta: {beta.reshape(-1)}, loss: {J(beta)}

while (np.abs(grad := gradient(beta)) > 1e-10).any() and (iter := iter - 1) > 0:
    # 计算梯度
    # grad  = gradient(beta)
    beta0.append(beta[0][0])
    beta1.append(beta[1][0])
    # 更新参数
    beta = beta - alpha * grad
    # 每迭代10次打印一次当前参数和损失值
    if iter % 10 == 0:
        print(f"iteration {iter}, beta: {beta.reshape(-1)}, loss: {J(beta)}")

print(iter)
# 画图
import matplotlib.pyplot as plt

plt.plot(beta0, beta1)

plt.show()
