from random import random

import numpy as np


# Initialize Network
def initializeNetworkWeights(scale=0.01):
    W = [np.random.random((3, 8)), np.random.random((8, 3))]
    W = [(x - np.ones(x.shape) * 0.5) * scale for x in W]
    b = [np.random.random(3), np.random.random(8)]
    b = [(x - np.ones(x.shape) * 0.5) * scale for x in b]
    return W, b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidPrime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_half_sum_squared(X, Y):
    return np.linalg.norm(X - Y) ** 2 / 2


def forward(X, W, b):
    a = X
    z_list = []
    for w, b in zip(W, b):
        z = np.matmul(w, a.T).T + b
        z_list.append(z)
        a = sigmoid(z)
    return a, z_list


# Getting Deltas
def getDeltas(z_list, Y, W):
    delta_list = []
    for i in reversed(range(len(z_list))):
        if i == len(z_list) - 1:
            z = z_list[i]
            a = sigmoid(z)
            delta = -(Y - a) * sigmoidPrime(z)
            delta_list.append(delta)
        else:
            z = z_list[i]
            delta = np.matmul(delta_list[len(delta_list) - 1], W[i + 1]) * sigmoidPrime(z)
            # delta = np.matmul(W[i + 1].T, delta_list[len(delta_list) - 1]).T*sigmoidPrime(z)
            delta_list.append(delta)
    return delta_list


# Backpropagate
def backprop(X, Y, W, b, learning_rate=0.03, weightDecay=0.01):
    a, zlist = forward(X, W, b)
    delta_list = getDeltas(zlist, Y, W)
    # Dont average here!!
    # delta_list = [np.average(x, axis=0) for x in delta_list]
    delta_list = list(reversed(delta_list))
    a_list = [sigmoid(x) for x in zlist]
    a_list.insert(0, np.array(X))
    a_list = [x.reshape(x.shape[0], 1, x.shape[1]) for x in a_list]
    delta_list = [x.reshape(x.shape[0], x.shape[1], 1) for x in delta_list]
    for i in range(len(delta_list)):
        derivativeW = a_list[i] * delta_list[i]
        derivativeW = np.average(derivativeW, axis=0)
        # Average here!!
        W[i] = W[i] - learning_rate * derivativeW - W[i] * (weightDecay / len(X))
        b[i] = b[i] - learning_rate * np.average(delta_list[i], axis=0).reshape(-1)
    return W, b


# Batch Gradient Descent
def batch_GD(X, Y, epoch=20000, learning_rate=0.003, weightDecay=0.1, scale=0.1):
    W, b = initializeNetworkWeights(scale)
    print(W)
    print(b)
    a, zlist = forward(X, W, b)
    print(a)
    for i in range(epoch):
        W, b = backprop(np.copy(X), Y, W, b, learning_rate, weightDecay)
        if i % 100 == 0:
            print("hi")
        a, z_list = forward(X, W, b)
        print(a)
        print(mean_half_sum_squared(a, Y))
    return W, b


# Batch Gradient Descent
def stochastic_GD(X, Y, epoch=20000, learning_rate=0.003, weightDecay=0.01, scale=0.1, sample_size=1):
    W, b = initializeNetworkWeights(scale)

    print(W)
    print(b)
    a, zlist = forward(X, W, b)
    print(a)
    for i in range(epoch):
        random_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
        W, b = backprop(X[random_indices], Y[random_indices], W, b, learning_rate, weightDecay)
        a, z_list = forward(X, W, b)
        # print(a)
        print(mean_half_sum_squared(a, Y))
    return W, b

def accuracy(X, Y, W, b):
    a, zlist = forward(X, W, b)
    y_pred = np.argmax(a, axis=1)
    Y_true = np.argmax(Y, axis=1)
    TP = np.sum(y_pred == Y_true)
    return TP / len(Y)


if __name__ == '__main__':
    # Initiliaze the Data
    X = np.zeros((8, 8))
    for i in range(8):
        X[i, i] = 1
    Y = np.array(X)

    W, b = initializeNetworkWeights()
    W, b = stochastic_GD(X, Y, epoch=2000000, learning_rate=0.3, weightDecay=0.001, scale=0.3,sample_size=8)
    a, zlist = forward(X, W, b)
    print(np.argmax(a, axis=1))
    print(accuracy(X, Y, W, b))

