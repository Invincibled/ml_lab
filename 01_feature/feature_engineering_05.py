
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归模型, Lasso回归，岭回归
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

'''
1. 生成数据
2. 划分训练集和测试集
3. 定义模型
4. 训练模型
5. 预测
6. 评估模型

'''


# 1. 生成数据

X = np.linspace(-3, 3, 300).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 300).reshape(-1, 1)

print(X.shape, y.shape)


# 画出散点图(2x3子图)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0,0].scatter(X, y, c='y', alpha=0.6, s=10)
ax[0,1].scatter(X, y, c='y', alpha=0.6, s=10)
ax[0,2].scatter(X, y, c='y', alpha=0.6, s=10)

# 2. 划分训练集和测试集

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)


# 过拟合（20次多项式）
poly20 = PolynomialFeatures(degree=20)
x_train = poly20.fit_transform(trainX)
x_test = poly20.fit_transform(testX)

# 一、定义模型 不加正则化
model = LinearRegression()

model.fit(x_train, trainY)

#  预测结果，计算误差
y_pred = model.predict(x_test)
test_loss1 = mean_squared_error(testY, y_pred)

# 画出拟合曲线，并写出训练误差和测试误差
# 创建平滑的X轴用于绘图
ax[0,0].plot(X, model.predict(poly20.fit_transform(X)), 'r')
ax[0,0].text(-3, 1, f"测试误差 : {test_loss1:.4f}")

ax[1,0].bar(range(21), model.coef_.reshape(-1))



# 二、定义模型 加正则化
lasso = Lasso(alpha=0.01)

lasso.fit(x_train, trainY)

#  预测结果，计算误差
y_pred2 = lasso.predict(x_test)
test_loss2 = mean_squared_error(testY, y_pred2)

# 画出拟合曲线，并写出训练误差和测试误差
# 创建平滑的X轴用于绘图
ax[0,1].plot(X, lasso.predict(poly20.fit_transform(X)), 'r')
ax[0,1].text(-3, 1, f"测试误差 : {test_loss2:.4f}")

ax[1,1].bar(range(21), lasso.coef_.reshape(-1))

# 二、定义模型 加正则化
ridge = Ridge(alpha=1)

ridge.fit(x_train, trainY)

#  预测结果，计算误差
y_pred3 = ridge.predict(x_test)
test_loss3 = mean_squared_error(testY, y_pred3)

# 画出拟合曲线，并写出训练误差和测试误差
# 创建平滑的X轴用于绘图
ax[0,2].plot(X, ridge.predict(poly20.fit_transform(X)), 'r')
ax[0,2].text(-3, 1, f"测试误差 : {test_loss3:.4f}")

ax[1,2].bar(range(21), ridge.coef_.reshape(-1))


plt.show()