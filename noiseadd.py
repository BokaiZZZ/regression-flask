import numpy as np


def NoiseAdd(X, M):
    """
    This function is to add random noise to the data as a
    regularization method. Repeat M times and find the mean of
    the noise. Then use element-wise multiplication to update X.

    Parameters
    ----------
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where `n_samples` is the number of samples and
    `n_features` is the number of features.

    M: int. Monte Carlo replicates of generating random noise.

    Returns
    -------
    X_noise: feature matrix after noise addition.
    """
    noise_sum = np.zeros((X.shape[0], X.shape[1]))
    for i in range(M):
        noise = np.random.normal(1, 1, (X.shape[0], X.shape[1]))
        noise_sum = noise_sum + noise
    noise_add = noise_sum / M
    X_noise = np.multiply(noise_add, X)
    return X_noise