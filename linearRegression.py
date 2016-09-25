import numpy as np
import matplotlib.pyplot as plt

Q2data = 'Q2data.txt'


def loadDataSet(fileName):
    xVal_0, xVal_1, yVal = np.loadtxt(fileName, delimiter='\t', unpack=True)
    xVal = np.array([xVal_0, xVal_1]).T  # n * p
    plt.plot(xVal_1, yVal, 'ro')
    plt.xlabel('xVal')
    plt.ylabel('yVal')
    plt.savefig('plot_xVal_yVal')
    plt.show()
    return xVal, yVal


def standRegres(xVal, yVal):
    theta = np.array([1.0, 1.0])
    learningRate = 0.001
    gradient =  np.dot(np.dot(xVal.T, xVal), theta) - np.dot(xVal.T, yVal)
    while np.linalg.norm(gradient) > 0.01:
        theta -= learningRate * gradient
        gradient = np.dot(np.dot(xVal.T, xVal), theta) - np.dot(xVal.T, yVal)
    plt.plot(xVal.T[1], np.dot(xVal, theta), '-', xVal.T[1], yVal, 'o')
    plt.savefig('standRegres')
    plt.show()
    return theta


## standRegres with norm quation
def standRegres0(xVal, yVal):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(xVal.T, xVal)), xVal.T), yVal)
    return theta


def polyRegres(xVal, yVal):
    xVal_2 = np.square(xVal.T[1])
    xVal_2 = np.reshape(xVal_2, (len(xVal_2), 1))
    print xVal_2.shape
    xVal = np.append(xVal, xVal_2, axis=1)
    print xVal[0]
    theta = np.array([1.0, 1.0, 1.0])
    learningRate = 0.001
    gradient =  np.dot(np.dot(xVal.T, xVal), theta) - np.dot(xVal.T, yVal)
    while np.linalg.norm(gradient) > 0.01:
        theta -= learningRate * gradient
        gradient = np.dot(np.dot(xVal.T, xVal), theta) - np.dot(xVal.T, yVal)
    plt.plot(xVal.T[1], np.dot(xVal, theta), '-', xVal.T[1], yVal, 'o')
    plt.savefig('polyRegres')
    plt.show()
    print theta
    return theta


xVal, yVal = loadDataSet(Q2data)
standRegres(xVal, yVal)
polyRegres(xVal, yVal)

