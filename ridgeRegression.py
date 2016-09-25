import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RRdata = 'RRdata.txt'

def loadDataSet(fileName):
    xVal_0, xVal_1, xVal_2, yVal = np.loadtxt(fileName, delimiter=' ', unpack=True)
    xVal = np.array([xVal_0, xVal_1, xVal_2]).T  # n * p
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xVal_1, xVal_2, yVal)
    # plt.savefig('plot_xVal_yVal_3D')
    # plt.show()
    return xVal, yVal


def ridgeRegress(xVal, yVal, lamda=0):
    lambdaI = lamda * np.eye(3)
    lambdaI[1][1] = 0
    beta = np.dot(np.dot(np.linalg.inv(np.dot(xVal.T, xVal) + lambdaI), xVal.T), yVal)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x1 = np.arange(np.min(xVal[:, 1]), np.max(xVal[:, 1]), 0.1)
    # x2 = np.arange(np.min(xVal[:, 2]), np.max(xVal[:, 2]), 0.1)
    # x1, x2 = np.meshgrid(x1, x2)
    # Y = beta[0] + beta[1] * x1 + beta[2] * x2
    # cm = plt.cm.get_cmap('RdYlBu')
    # ax.plot_surface(x1, x2, Y, cmap=cm)
    # plt.hold(True)
    # ax.scatter(xVal.T[1], xVal.T[2], yVal, c='r')
    # plt.savefig('ridgeRegresL0')
    # plt.show()
    # print beta
    return beta

def cv(xVal, yVal):
    index = range(0, len(yVal))
    random.seed(37)
    random.shuffle(index)
    x = xVal
    y = yVal
    for i in range(len(yVal)):
        x[i] = xVal[index[i]]
        y[i] = yVal[index[i]]  # get shuffled x and y
    lambdas = []
    for i in range(50):
        lambdas.append(0.02 + i * 0.02)
    mse = []
    fold_size = len(yVal) / 10
    for i in range(len(lambdas)):
        error = 0
        for j in range(10):
            test_x = x[j * fold_size: (j + 1) * fold_size]
            train_x = np.append(x[0: j * fold_size], x[(j + 1) * fold_size:], axis=0)
            test_y = y[j * fold_size: (j + 1) * fold_size]
            train_y = np.append(y[0: j * fold_size], y[(j + 1) * fold_size:], axis=0)
            beta = ridgeRegress(train_x, train_y, lambdas[i])
            error += np.sum(np.square(test_y - np.dot(test_x, beta)))/fold_size
        mse.append(error / 10)
    print min(mse)
    print mse.index(min(mse))
    print lambdas[mse.index(min(mse))]



xVal, yVal = loadDataSet(RRdata)
# ridgeRegress(xVal, yVal)
cv(xVal, yVal)