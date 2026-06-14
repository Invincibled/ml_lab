from sklearn.neighbors import KNeighborsRegressor

X = [[2,1],[3,1],[1,4],[2,6]]  # 数据准备
y = [0.5,0.33,4,3] # 分类标签

knn = KNeighborsRegressor(n_neighbors=2, weights='distance')

knn.fit(X,y)
x = [[4,9]]
x_pred = knn.predict(x)
print(x_pred)
