import sys, os, pickle
sys.path.append('C:\\Users\\alesi\\PycharmProjects\\DeepLeaning\\deep-learning-from-scratch-master')
from dataset.mnist import load_mnist
import numpy as np

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("C:\\Users\\alesi\\PycharmProjects\\DeepLeaning\\deep-learning-from-scratch-master\\ch03\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ソフトマックス関数
def softmax(a):
    max_a = np.max(a)
    exp_a = np.exp(a - max_a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 隠れ層2つのNN
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    Z1 = sigmoid(np.dot(x, W1) + b1)
    Z2 = sigmoid(np.dot(Z1, W2) + b2)
    Z3 = softmax(np.dot(Z2, W3) + b3)

    return Z3

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if i % 1000 == 0:
        print("Predict:" + str(float(p)) + "  Test:" + str(float(t[i])))
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
