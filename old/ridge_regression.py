
import numpy as np
import numexpr as ne

class RidgeRegression:
    # parameters
    llambda: float
    sigma: float
    length_scale: float

    # attributes
    X: np.ndarray
    alpha: np.ndarray
    K: np.ndarray

    def __init__(self, llambda=0.0001, sigma=2, length_scale=1):
        self.llambda = llambda
        self.sigma = sigma
        self.length_scale = length_scale
        if length_scale == 0:
            self.length_scale = 0.0000001

    def gaussian_kernel(self, X: np.ndarray, Y: np.ndarray):
        def k(x, y):
            return np.exp(- (np.linalg.norm(x - y) ** 2) / 2 * self.sigma ** 2)

        K = np.ndarray((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                K[i][j] = k(x, y)
        return K

    def rbf_fast(self, X: np.ndarray, Y:np.ndarray):
        X_norm = np.einsum('ij,ij->i', X, X)
        if Y is not None:
            Y_norm = np.einsum('ij,ij->i', Y, Y)
        else:
            Y = X
            Y_norm = X_norm

        K = ne.evaluate('v * exp(-g * (A + B - 2 * C))', {
            'A': X_norm[:, None],
            'B': Y_norm[None, :],
            'C': np.dot(X, Y.T),
            'g': 1 / (2 * self.length_scale ** 2),
            'v': self.sigma
        })
        return K

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        n_samples = X.shape[0]
        self.K = self.rbf_fast(X, X)
        inv = np.linalg.inv(self.K + self.llambda * np.identity(n_samples))
        self.alpha = np.dot(inv, y.T)

    def predict(self, x_new):
        assert x_new.ndim == self.X.ndim
        return np.dot(self.alpha.T, self.rbf_fast(self.X, x_new))
