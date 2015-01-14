import numpy as np
import scipy.sparse as sp


def rmse(X_true, X_pred):
    if X_true.shape != X_pred.shape:
        raise ValueError("X_true and X_pred should have the same shape.")

    # FIXME: we just need to check that both matrix
    # have the same sparse format.
    X_true = sp.csr_matrix(X_true)
    X_pred = sp.csr_matrix(X_pred)

    mse = np.mean((X_true.data - X_pred.data) ** 2)
    return np.sqrt(mse)
