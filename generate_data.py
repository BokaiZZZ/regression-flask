import numpy as np


def generate_data(p=1, sampleSize=100, rand=1):
    X = np.random.randn(sampleSize, p)
    slope = np.random.randint(-5, 5, p).T
    intercept = np.random.randint(-10, 10)
    y = X@slope + intercept + np.random.normal(0.0, rand, size=sampleSize)
    return X, y
