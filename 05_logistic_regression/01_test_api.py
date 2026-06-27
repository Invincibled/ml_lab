from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver='liblinear',
    multi_class='multinomial',
    C=1.0,
)

# Softmax回归
model_softmax= LogisticRegression(multi_class='multinomial')
# 对于多分类问题，LogisticRegression会自动使用multinomial，因此multi_class参数可省略