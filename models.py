import numpy as np
#from sklearn.linear_model import LinearRegression

class LogisticRegression:
    def __init__(self, eta, threshold, num_iters):
        self.theta = None
        self.eta = eta
        self.threshold = threshold
        self.num_iters = num_iters
        self.sigmoid = np.vectorize(lambda x : 1 / (1 + np.exp(-x)))
        self.ll = np.zeros(num_iters)
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iters):
            yHat = self.sigmoid(np.matmul(X, self.theta))
            self.ll[i] = np.dot(y, np.log(yHat)) + np.dot((1 - y), np.log(1 - yHat))
            if i != 0 and self.ll[i] - self.ll[i-1] <= self.threshold: break
            coeff = y - yHat
            grad = np.sum(np.multiply(X, coeff[:, None]), axis=0)
            self.theta += self.eta * grad
    
    def predict(self, X):
        f = np.vectorize(lambda x : 1.0 if x > 0.5 else 0.0)
        yHat = self.sigmoid(np.matmul(X, self.theta))
        return f(yHat)
    
# TODO: make random initializations to choose best accuracy
class LinearRegression:
    def __init__(self, eta, threshold, num_iters):
        self.theta = None
        self.eta = eta
        self.threshold = threshold
        self.num_iters = num_iters
        self.sl = np.zeros(num_iters)
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1]) # fuck with this line, ones > zeros, but use random
        square = np.vectorize(lambda x : x**2)
        for i in range(self.num_iters):
            print(self.theta)
            yHat = np.matmul(X, self.theta)
            self.sl[i] = sum(square(yHat - y))
            if i != 0 and self.sl[i] - self.sl[i-1] <= self.threshold: break
            coeff = 2*(y - yHat)
            grad = np.sum(np.multiply(X, coeff[:, None]), axis=0) *  (1/X.shape[1])
            self.theta += self.eta * grad
    
    def predict(self, X):
        return np.matmul(X, self.theta)
    
# TODO: make random initializations to choose best accuracy
class LassoRegression:
    def __init__(self, eta, threshold, num_iters, alpha):
        self.theta = None
        self.eta = eta
        self.threshold = threshold
        self.num_iters = num_iters
        self.sl = np.zeros(num_iters)
        self.alpha = alpha
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1]) # fuck with this line, ones > zeros, but use random
        square = np.vectorize(lambda x : x**2)
        for i in range(self.num_iters):
            print(self.theta)
            yHat = np.matmul(X, self.theta)
            self.sl[i] = sum(square(yHat - y))
            if i != 0 and self.sl[i] - self.sl[i-1] <= self.threshold: break
            coeff = 2 * (y - yHat)
            grad = np.sum(np.multiply(X, coeff[:, None]) + self.alpha * np.linalg.norm(self.theta, ord=1), axis=0) *  (1/X.shape[1]) # CHANGED THIS LINE
            self.theta += self.eta * grad
    
    def predict(self, X):
        return np.matmul(X, self.theta)
    
class RidgeRegression:
    def __init__(self, eta, threshold, num_iters, alpha):
        self.theta = None
        self.eta = eta
        self.threshold = threshold
        self.num_iters = num_iters
        self.sl = np.zeros(num_iters)
        self.alpha = alpha
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1]) # fuck with this line, ones > zeros, but use random
        square = np.vectorize(lambda x : x**2)
        for i in range(self.num_iters):
            print(self.theta)
            yHat = np.matmul(X, self.theta)
            self.sl[i] = sum(square(yHat - y))
            if i != 0 and self.sl[i] - self.sl[i-1] <= self.threshold: break
            coeff = 2 * (y - yHat)
            grad = np.sum(np.multiply(X, coeff[:, None]) + 2 * self.alpha * np.linalg.norm(self.theta, ord=2), axis=0) * (1/X.shape[1]) # CHANGED THIS LINE
            self.theta += self.eta * grad
    
    def predict(self, X):
        return np.matmul(X, self.theta)
