import numpy as np
import matplotlib.pyplot as plt

Q2data = 'Q2data.txt'
RRdata = 'RRdata.txt'


def loadDataSet(fileName):
    xVal_0, xVal_1, yVal = np.loadtxt(fileName, delimiter='\t', unpack=True)
    xVal = np.array([xVal_0, xVal_1]).T  # p *  n
    plt.plot(xVal_1, yVal, 'ro')
    plt.xlabel('xVal')
    plt.ylabel('yVal')
    plt.savefig('plot_xVal_yVal')
    plt.show()
    print xVal.shape
    return xVal, yVal


def standRegres(xVal, yVal):
    theta_old = np.array([1, 1])
    theta_new = theta_old
    learningRate = 0.01
    sse_old = np.sum(np.square(np.subtract(np.dot(xVal, theta_old), yVal)))
    while sse_old > 0.001:
        theta_new = theta_old + learningRate * np.dot(xVal.T, yVal - np.dot(xVal, theta_old))
        print theta_new
        sse_new = np.sum(np.square(np.subtract(np.dot(xVal, theta_new), yVal)))
        print sse_new
        if sse_new > sse_old and False:
            break
        else:
            sse_old = sse_new
            theta_old = theta_new
    theta = theta_old
    print theta
    plt.plot(xVal[1], np.dot(xVal, theta), '-', xVal[1], yVal, 'o')
    plt.show()

def standRegres0(xVal, yVal):
    theta = np.dot(np.dot(np.linalg.inv(np.dot(xVal.T, xVal)), xVal.T), yVal)
    # theta[1] = np.dot(np.linalg.inv(np.dot(xVal.T, xVal)), xVal.T)[1] * yVal
    print theta

xVal, yVal = loadDataSet(Q2data)
standRegres(xVal, yVal)
