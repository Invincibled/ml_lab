import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数
def J(x):
    return (x**2 - 2) ** 2

# 定义目标函数的梯度(导函数)
def gradient(x):
    return 4*x**3 - 8*x


x_list = []
y_list = []


# 定义超参数和x的初始值
alpha = 0.1
x = 1


# 重复迭代100次
while np.abs((grad := gradient(x))) > 1e-10:
    y = J(x)
    x_list.append(x)
    y_list.append(y)
    print(f"Iteration x = {x}, y = {y}")
    x = x - alpha * grad

print(len(x_list))

# x =np.arange(0.9, 1.6, 0.01)
# plt.plot(x, J(x))
# plt.plot(x_list, y_list, 'ro')
# plt.scatter(x_list, y_list, color='red')
# plt.show()


#  局部放大
fig, ax = plt.subplots(1,2, figsize=(15,4))
x = np.arange(0.9, 1.6, 0.01)
ax[0].plot(x, J(x))
ax[0].plot(x_list, y_list, 'r')
ax[0].scatter(x_list, y_list, color='r')

x_list2= x_list[1:]
y_list2= y_list[1:]
x = np.arange(1.399, 1.425, 0.001)
ax[1].plot(x, J(x))
ax[1].plot(x_list2, y_list2, 'r')
ax[1].scatter(x_list2, y_list2, color='r')

plt.show()




