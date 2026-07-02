"""
SVD (奇异值分解) 示例
=======================
SVD 将任意 m×n 矩阵 A 分解为:
    A = U Σ V^T
其中:
    U : m×m 正交矩阵, 列向量称为左奇异向量
    Σ : m×n 对角矩阵, 对角线上的非负实数称为奇异值 (降序排列)
    V^T: n×n 正交矩阵, 行向量称为右奇异向量

本示例演示:
1. 对一个 4×2 矩阵进行 SVD 分解
2. 用分解结果重构原矩阵, 验证 A ≈ U Σ V^T
3. 利用截断 SVD 进行降维 / 近似 (低秩近似)
4. 从几何角度观察 U、V 的作用
"""
import numpy as np

# ---------- 1. 构造一个矩阵 ----------
# 4 个样本, 2 个特征
# 行间相关性较强, 容易看出低秩近似的效果
A = np.array([
    [1.0, 2.0],
    [2.0, 4.1],
    [3.0, 6.0],
    [4.0, 8.2],
])
print("=" * 50)
print("原始矩阵 A (shape = {})".format(A.shape))
print(A)

# ---------- 2. SVD 分解 ----------
# full_matrices=False: 返回精简形式, U 为 m×k, Σ 为 k, Vt 为 k×n, k=min(m,n)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print("\n" + "=" * 50)
print("SVD 分解结果")
print("-" * 50)
print("U (左奇异向量, shape = {}):".format(U.shape))
print(U)
print("\ns (奇异值, 降序排列, shape = {}):".format(s.shape))
print(s)
print("\nVt (右奇异向量, V^T, shape = {}):".format(Vt.shape))
print(Vt)

# ---------- 3. 验证: 用 U, s, Vt 重构 A ----------
# Σ 是对角矩阵, 需要把奇异值向量 s 还原成对角矩阵
Sigma = np.diag(s)
A_reconstructed = U @ Sigma @ Vt
print("\n" + "=" * 50)
print("重构矩阵 A_reconstructed = U @ Σ @ V^T")
print(A_reconstructed)
print("\n重构误差 max|A - A_reconstructed| = {:.2e}".format(np.max(np.abs(A - A_reconstructed))))

# ---------- 4. 截断 SVD: 低秩近似 ----------
# 只保留最大的奇异值, 进行秩 1 近似 (数据共 2 个奇异值, 砍掉较小的那个)
r = 1
U_r = U[:, :r]            # 取前 r 列
s_r = s[:r]               # 取前 r 个奇异值
Vt_r = Vt[:r, :]          # 取前 r 行
A_approx = U_r @ np.diag(s_r) @ Vt_r

print("\n" + "=" * 50)
print("秩-{} 近似 (保留前 {} 个奇异值)".format(r, r))
print("保留的奇异值:", s_r)
print("近似矩阵 A_approx:")
print(A_approx)
print("\n近似误差 max|A - A_approx| = {:.4f}".format(np.max(np.abs(A - A_approx))))
print("近似误差 Frobenius 范数 = {:.4f}".format(np.linalg.norm(A - A_approx, ord='fro')))

# ---------- 5. 信息保留率 ----------
# 被截断的奇异值平方和 占比 越小, 信息保留越完整
total_energy = np.sum(s ** 2)
kept_energy = np.sum(s_r ** 2)
print("\n信息保留率 = {:.2%}".format(kept_energy / total_energy))
print("  => 含义: 用 {} 个分量就保留了 {:.2f}% 的数据信息".format(r, 100 * kept_energy / total_energy))

# ---------- 6. 几何意义 ----------
# V 的行向量构成一组新的正交基; A @ V^T 把数据投影到这组新基上
# U @ Σ 则是投影后的坐标 (得分)
print("\n" + "=" * 50)
print("几何意义")
print("-" * 50)
print("右奇异向量 V (主方向):")
print(Vt.T)  # V = Vt^T
print("\n将原始数据投影到主方向得到的得分 (A @ V = U @ Σ):")
scores = A @ Vt.T
print(scores)
print("\nU @ Σ (应与上面相等):")
print(U @ np.diag(s))
