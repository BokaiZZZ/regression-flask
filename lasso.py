import numpy as np
from regression import RegressionAlgorithm


class Lasso(RegressionAlgorithm):
    """
    Linear Model trained with L1 prior as regularizer (Lasso).
    Use gradient descent to find the optimize value.
    """

    def __init__(self):
        super(Lasso, self).__init__()
        self.robust = False

    def gradient(self):
        """
        Return the gradient for each step. If robust is true, a regularizer is added
        """
        m = self.X.shape[0]
        if self.robust:
            if len(self.c) != self.X.shape[1]:
                self.c = np.insert(self.c, 0, values=np.ones(1), axis=0)
            return (1 / m) * (self.X.T @ (self.h() - self.y)
                              + 0.5 * (self.c + self.alpha) @ np.sign(self.beta))
        else:
            return (1 / m) * (self.X.T @ (self.h() - self.y)
                              + 0.5 * self.alpha @ np.sign(self.beta))

    def fit(self, X, y, **kwargs):
        """
        fit the X,y to find the beta with fixed learning rate and number of epochs
        """
        self.X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)
        self.y = y.reshape(-1, 1)
        alpha = np.ones((self.X.shape[1]))
        for key, value in kwargs.items():
            if key == "alpha":
                alpha = value
            elif key == "c":
                self.c = value

        self.alpha = np.multiply(alpha, np.ones((self.X.shape[1])))
        self.beta = np.random.rand(self.X.shape[1]).reshape(-1, 1)

        num_epochs = 10000
        for _ in range(num_epochs):
            self.beta = self.beta - 0.1 * self.gradient()
