import numpy as np
from dropout import dropout
from noiseadd import NoiseAdd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error


def regression(model, X, y, **kwargs):
    """
    This function can automatically tune blackbox regression models.

    Parameters
    ----------
    model: regression algorithm model object

    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y: array-like of shape (n_samples,) Target values.

    method: {'Dropout', 'NoiseAddition', 'Robust'}

    M: int of Monte Carlo replicates for Dropout or NoiseAddition methods.

    c: column bounds to be used if the method specified is Robust.

    K: int of CV-folds to be used to tune the amount of regularization.

    alpha: float, default=1.0. Regularization strength; must be a positive float.
           Used when model is Lasso or Ridge.

    criterion: {'MSE', 'MAD'}. Strategy to evaluate the performance of the
    cross-validated model on the test set. Especially MSE (mean square error) and
    MAD (mean absolute deviation) in this function.

    Returns
    -------
    best_model : trained best model after regularization and cross validation.
    """
    noise = False
    drop = False
    K = 5
    criterion = "MSE"
    # read kwargs
    for key, value in kwargs.items():
        if key == "method":
            if value == "Robust":
                model.robust = True
            elif value == "Dropout":
                drop = True
            elif value == "NoiseAddition":
                noise = True
        elif key == "M":
            M = value
        elif key == "K":
            K = value
        elif key == "criterion":
            criterion = value

    # Regularization
    if drop:
        X = dropout(X, M)
    if noise:
        X = NoiseAdd(X, M)

    models = []
    cv_scores = []
    # CV based on criterion
    kf = KFold(n_splits=K, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train, **kwargs)
        y_pred = model.predict(X_test)
        if criterion == "MAD":
            score = mean_absolute_error(y_test, y_pred)
        elif criterion == "MSE":
            score = mean_squared_error(y_test, y_pred)
        cv_scores.append(score)
        models.append(model)
    best_idx = np.argmin(cv_scores)
    best_model = models[best_idx]
    return best_model
