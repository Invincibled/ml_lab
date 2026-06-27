# 实现与门
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    linear_combination = w1 * x1 + w2 * x2
    if linear_combination >= theta:
        return 1
    else:
        return 0
print("AND(0, 0)", AND(0, 0))
print("AND(0, 1)", AND(0, 1))
print("AND(1, 0)", AND(1, 0))
print("AND(1, 1)", AND(1, 1))

# 重新实现与门，使用矩阵
import numpy as np
def AND_matrix(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    theta = 0.7
    linear_combination = np.dot(x, w)
    if linear_combination >= theta:
        return 1
    else:
        return 0
print("AND(0, 0)", AND_matrix(0, 0))
print("AND(0, 1)", AND_matrix(0, 1))
print("AND(1, 0)", AND_matrix(1, 0))
print("AND(1, 1)", AND_matrix(1, 1))

# 实现或门
def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.2
    linear_combination = w1 * x1 + w2 * x2
    if linear_combination >= theta:
        return 1
    else:
        return 0
print("OR(0, 0)", OR(0, 0))
print("OR(0, 1)", OR(0, 1))
print("OR(1, 0)", OR(1, 0))
print("OR(1, 1)", OR(1, 1))


# 多层感知机，异或门，给出python代码，不要用上面代码，自己实现
# 异或门 XOR 真值表:
#   x1 x2 | y
#    0  0 | 0
#    0  1 | 1
#    1  0 | 1
#    1  1 | 0
# 异或门无法用单层感知机实现(非线性可分), 需要使用多层感知机(两层结构)
# 思路: XOR(x1, x2) = AND(OR(x1, x2), NAND(x1, x2))
#   第一层: OR 门 和 NAND 门
#   第二层: AND 门

def _step(x):
    """阶跃激活函数"""
    return 1 if x >= 0 else 0

def _perceptron(x1, x2, w1, w2, b):
    """通用单层感知机: w1*x1 + w2*x2 + b, 经阶跃函数激活"""
    return _step(w1 * x1 + w2 * x2 + b)

def OR_gate(x1, x2):
    """或门: 权重 1, 1, 偏置 -0.5 -> x1+x2 >= 0.5 时输出 1"""
    return _perceptron(x1, x2, 1.0, 1.0, -0.5)

def NAND_gate(x1, x2):
    """与非门: 权重 -1, -1, 偏置 1.5 -> x1+x2 <= 1.5 时输出 1(即不同时为1)"""
    return _perceptron(x1, x2, -1.0, -1.0, 1.5)

def AND_gate(x1, x2):
    """与门: 权重 1, 1, 偏置 -1.5 -> x1+x2 >= 1.5 时输出 1"""
    return _perceptron(x1, x2, 1.0, 1.0, -1.5)

def XOR(x1, x2):
    """异或门(多层感知机实现):
       第一层: s1 = OR(x1, x2), s2 = NAND(x1, x2)
       第二层: y = AND(s1, s2)
    """
    s1 = OR_gate(x1, x2)
    s2 = NAND_gate(x1, x2)
    y = AND_gate(s1, s2)
    return y

# 测试异或门
print("XOR(0, 0) =", XOR(0, 0))  # 期望 0
print("XOR(0, 1) =", XOR(0, 1))  # 期望 1
print("XOR(1, 0) =", XOR(1, 0))  # 期望 1
print("XOR(1, 1) =", XOR(1, 1))  # 期望 0



