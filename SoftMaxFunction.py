import numpy as np

# オーバーフローを考慮しないソフトマックス関数
def softmax1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# オーバーフローを考慮したソフトマックス関数
# 入力値の最大値を各入力値から引いて計算
def softmax2(a):
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
print(softmax1(a))

b = np.array([1010, 1000, 990])
# overflow
#print(softmax1(b))
print(softmax2(b))

# aについてもsoftmax2()を利用
print(softmax2(a))
