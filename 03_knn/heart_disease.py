import os

import pandas as pd
from Crypto.SelfTest.Cipher.test_CFB import file_name
from sklearn.model_selection import train_test_split
from xgboost.testing.data import joblib

# 加载数据集
# heart_disease_data = pd.read_csv('../data/heart_disease.csv')
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建 data 目录的绝对路径
data_path = os.path.join(current_dir, '..', 'data', 'heart_disease.csv')
heart_disease_data = pd.read_csv(data_path, encoding='utf-8')# 处理缺失值.
heart_disease_data.dropna(inplace=True)

heart_disease_data.info()
print(heart_disease_data.head())


X = heart_disease_data.drop("是否患有心脏病", axis=1)
y = heart_disease_data["是否患有心脏病"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
'''
特征转换
数据集中包含多种类型的特征:
类别型特征(需要特殊处理)
胸痛类型:4种分类(名义变量)
静息心电图结果:3种分类(名义变量)
峰值ST段的斜率:3种分类(有序变量)
地中海贫血:4种分类(名义变量)
数值型特征(可直接标准化):年龄、静息血压、胆固醇、最大心率、运动后的ST
下降、主血管数量
二元特征(保持原样):性别、空腹血糖、运动性心绞痛
对于类别型特征,直接使用整数编码的类别特征会被算法视为有序数值,导致错误的距
离计算(例如:会认为胸痛类型=1和胸痛类型=2之间的差异比胸痛类型=1和胸痛类
型=3之间差异更小,而实际上它们都是类别)。使用独热编码(One-HotEncoding)
可将类别特征转换为二元向量，消除虚假的顺序关系。
'''

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# 特征工程
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
# print(X_train.shape, X_test.shape)
#
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier(n_neighbors=3)
#
# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)
#
# print(y_pred)
# # 模型评估
# score = knn.score(X_test, y_test)
# print(score)

# 保存模型
# joblib.dump(value=knn, filename='knn_model.pkl')
# print("模型已保存为 knn_model.pkl")


# 加载模型
knn = joblib.load(filename='knn_model.pkl')
print("模型已加载")

y_pred = knn.predict(X_test[10:11])
print(f"预测结果为：{y_pred}，真实结果为：{y_test[10]}")

