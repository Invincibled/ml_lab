import numpy as np
from sklearn.linear_model import LinearRegression

# 自变量,每周学习时长
X = [[5], [8], [10], [12], [15], [3], [7], [9], [14],[6]]
#因变量,数学考试成绩
y = [55,65,70,75,85,50,60,72,80,58]

# model = LinearRegression()
# model.fit(X, y)
# print(model.coef_)
# print(model.intercept_)
# print(model.predict([[10]]))


# 画图
# import matplotlib.pyplot as plt
# plt.scatter(X, y)
# plt.plot(X, model.predict(X), color='red')
# plt.show()

# x_line = np.arange(0, 15, 0.1).reshape(-1, 1)
# y_line = model.predict(x_line)
# import matplotlib.pyplot as plt
# plt.scatter(X, y)
# plt.plot(x_line, y_line, color='red')
# plt.show()

# 定义模型, 截距为false，即不做截距.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)
print(model.coef_)
print(model.intercept_)




