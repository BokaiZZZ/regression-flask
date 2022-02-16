from lasso import Lasso
import numpy as np


class Ridge(Lasso):
    """
    Linear least squares with l2 regularization (Ridge).
    Use gradient descent to find the optimize value.
    """

    def __init__(self):
        super(Ridge, self).__init__()
        self.robust = False

    def gradient(self):
        """
        Return the gradient for each step. If robust is true, a regularizer is added
        """
        m = self.X.shape[0]
        if self.robust:
            if len(self.c) != self.X.shape[1]:
                self.c = np.insert(self.c, 0, values=np.ones(1), axis=0)
            return (1 / m) * (self.X.T @ (self.h() - self.y) +
                              + self.alpha @ self.beta + 0.5 * self.c @ np.sign(self.beta))
        else:
            return (1 / m) * self.X.T @ (self.h() - self.y) + self.alpha @ self.beta
