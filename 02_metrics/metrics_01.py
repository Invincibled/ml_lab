import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
# 定义类别标签
labels =["猫","狗"]
# 定义数据（预测值和真实值）
y_true = ["猫","猫","猫","猫","猫","猫","狗","狗","狗","狗"]
y_pred = ["猫","猫","狗","猫","猫","猫","猫","猫","狗","狗"]

# 得到混淆矩阵
matrix = confusion_matrix(y_true, y_pred)
print(matrix)
# sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="猫")
recall = recall_score(y_true, y_pred, pos_label="猫")
f1 = f1_score(y_true, y_pred, pos_label="猫")
report = classification_report(y_true, y_pred, labels=labels, target_names=None)

print(report)


print(accuracy)

print(precision)
print(f1)
print(recall)
