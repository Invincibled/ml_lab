import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# 1. 加载数据集
data = pd.read_csv('../data/heart_disease.csv')
data.dropna(inplace=True)

# 2. 划分数据集
X = data.drop(columns=['是否患有心脏病'], axis=1)
y = data['是否患有心脏病']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 3. 特征工程
# 数值型特征
numerical_features=["年龄","静息血压","胆固醇","最大心率","运动后的ST下降","主血管数量"]
#类别型特征
categorical_features=["胸痛类型","静息心电图结果","峰值ST段的斜率","地中海贫血"]
#二元特征
binary_features=["性别","空腹血糖","运动性心绞痛"]

# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

# 特征转换
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print(X_train.shape, X_test.shape)
# 4. 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. 模型评估
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)
y_pred = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))





