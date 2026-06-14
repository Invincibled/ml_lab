import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数
def f(x):
    return x**2

# 定义目标函数的梯度(导函数)
def gradient(x):
    return 2*x


x_list = []
y_list = []


# 定义超参数和x的初始值
alpha = 0.01
x = 1


# 重复迭代100次
for i in range(100):
    x_list.append(x)
    y_list.append(f(x))
    print(f"Iteration {i+1}: x = {x}, y = {f(x)}")
    grad = gradient(x)
    x = x - alpha * grad


x =np.arange(-1, 1, 0.01)
plt.plot(x, f(x))
plt.plot(x_list, y_list, 'ro')
plt.scatter(x_list, y_list, color='red')
plt.show()


