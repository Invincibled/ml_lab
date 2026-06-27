import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error # 均方误差


# 1. 读取数据
dataset = pd.read_csv('../data/data_advertising.csv')
dataset.dropna(inplace=True)
dataset.drop(columns=dataset.columns[0], axis=1, inplace=True)

dataset.info()
print(dataset.head())

# 2. 划分数据集
X = dataset.drop(columns=['sales'], axis=1)
y = dataset['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 3. 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 训练模型
model = LinearRegression()
model.fit(X_train, y_train)
print("LR coef_: ", model.coef_)
print("LR intercept_: ", model.intercept_)
# 5. 预测
y_pred = model.predict(X_test)
# 6. 评估模型
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)


# 4.2 随机梯度下降法
sgd_model = SGDRegressor()
sgd_model.fit(X_train, y_train)
print("SGD coef_: ", sgd_model.coef_)
print("SGD intercept_: ", sgd_model.intercept_)
y_pred_sgd = sgd_model.predict(X_test)
mse_sgd = mean_squared_error(y_test, y_pred_sgd )
print('Mean Squared Error:', mse_sgd)
