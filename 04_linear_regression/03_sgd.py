
import numpy as np
# 导入sgd
from sklearn.linear_model import SGDRegressor

# 1. 定义数据
X = np.array([[5], [8], [10], [12], [15], [3], [7],[9], [14], [6]])
# 自变量,每周学习时长
y = np.array([[55], [65], [70], [75], [85], [50], [60],[72],[80], [58]]) #因变量,数学考试成绩

sgd = SGDRegressor(
    penalty=None,
    loss='squared_error',
    max_iter=10**7,
    eta0=1e-7,
    learning_rate='constant',
    tol=1e-8
)
sgd.fit(X, y)
print(sgd.coef_)
print(sgd.intercept_)
