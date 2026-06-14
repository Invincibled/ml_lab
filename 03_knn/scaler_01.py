
from sklearn.preprocessing import MinMaxScaler

X = [[2,1],[3,1],[1,4],[2,6]]
scaler = MinMaxScaler(feature_range=(-1,1))
X_scaled = scaler.fit_transform(X)
print(X_scaled)



from sklearn.preprocessing import StandardScaler
X = [[2,1],[3,1],[1,4],[2,6]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
