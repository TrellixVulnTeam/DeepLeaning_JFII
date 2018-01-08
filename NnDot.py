import numpy as np

# 重み(W)とバイアス(b)の設定
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# 出力層は恒等関数
def identity_function(x):
    return x

# 3層ニューラルネットワークの計算
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    Z1 = sigmoid(np.dot(x, W1) + b1)
    Z2 = sigmoid(np.dot(Z1, W2) + b2)
    Z3 = identity_function(np.dot(Z2, W3) + b3)
    return Z3

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# メイン処理
# 重みとバイアスを初期化後、入力層として(1.0, 0.5)を与える
# 2層の隠れ層を経て出力層に至る
# 隠れ層の活性化関数はシグモイド関数を利用
# 出力層の活性化関数は恒等関数を利用
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
