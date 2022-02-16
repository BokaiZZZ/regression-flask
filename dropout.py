import numpy as np
import random


def dropout(X, M):
    """
    This function is to randomly drop some features in X as a
    regularization method. Find M dropout positions randomly.
    Then use element-wise multiplication to update X.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where `n_samples` is the number of samples and
    `n_features` is the number of features.

    M: int. Number of randomly selected features to be ignored.

    Returns
    -------
    X_drop: feature matrix after dropout.
    """
    dropout_mtr = np.ones(X.size)
    dropout_idx = random.sample(range(0, X.size - 1), M)
    dropout_mtr[dropout_idx] = 0
    dropout_mtr = dropout_mtr.reshape(X.shape[0], X.shape[1])
    X_drop = np.multiply(dropout_mtr, X)
    return X_drop
