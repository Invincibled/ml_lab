import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# 1. 加载数据集
dataset = pd.read_csv('../data/digit-recognizer/train.csv')
print(dataset.shape)
print(dataset.head())
# print(dataset.describe())

# 2. 测试图像
# test_image = dataset.iloc[10,1:].values
# plt.imshow(test_image.reshape(28,28), cmap='gray')
# plt.show()

# 3. 划分训练集和测试集
X = dataset.drop(['label'], axis=1)
y = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 4. 特征工程，归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5.定义模型和训练
model = LogisticRegression(max_iter=500)

model.fit(X_train, y_train)
# 6.预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# 7. 测试(预测某个图片表示的数字)
digit = X_test[123, :].reshape(1, -1)
print("Predicted digit:", model.predict(digit))
plt.imshow(X_test[123, :].reshape(28,28), cmap='gray')
plt.show()








