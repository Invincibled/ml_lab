# 分类任务

from sklearn.datasets import make_classification # 能够自动生成分类任务数据集
from sklearn.metrics import classification_report # 分类报告
from sklearn.model_selection import train_test_split # 数据集分割
from sklearn.linear_model import LogisticRegression # 逻辑回归模型

X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X)
print(y)
# 逻辑回归模型
model = LogisticRegression()
# 训练模型
model.fit(X_train, y_train)
# 预测 测试
y_pred = model.predict(X_test)
# 分类报告
report = classification_report(y_test, y_pred)
print(report)

# 获取预测正类的概率值
y_pred_proba = model.predict_proba(X_test)[:,1]
print(y_pred_proba)

# 计算AUC值
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred_proba)
print(auc)
