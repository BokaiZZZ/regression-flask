import numpy as np


class RegressionAlgorithm(object):
    """
    Ordinary least squares Linear Regression.
    Use gradient descent to find the optimizing value.
    Predicted coefficients are stored by self.beta after self.fit(X,y)
    """

    def __init__(self):
        self.robust = False

    def h(self):
        return np.matmul(self.X, self.beta)

    def gradient(self):
        """
        Return the gradient for each step. If robust is true, a regularizer is added
        """
        m = self.X.shape[0]
        if self.robust:
            if len(self.c) != self.X.shape[1]:
                self.c = np.insert(self.c, 0, values=np.ones(1), axis=0)
            return (1 / m) * (self.X.T @ (self.h() - self.y) + self.c @ np.sign(self.beta))
        else:
            return (1 / m) * self.X.T @ (self.h() - self.y)

    def fit(self, X, y, **kwargs):
        """
        fit the X,y to find the beta with fixed learning rate and number of epochs
        """
        self.X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        self.y = y.reshape(-1, 1)
        for key, value in kwargs.items():
            if key == "c":
                self.c = value
        self.beta = np.random.rand(self.X.shape[1]).reshape(-1, 1)
        num_epochs = 10000
        for _ in range(num_epochs):
            self.beta = self.beta - 0.1 * self.gradient()

    def predict(self, X_test):
        return np.insert(X_test, 0, values=np.ones(X_test.shape[0]), axis=1) @ self.beta
