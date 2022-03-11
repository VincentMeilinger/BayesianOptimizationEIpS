import numpy as np
import numexpr as ne
import scipy

class BayesianOptimizer:

    # parameters
    sigma: float
    noise: float
    length_scale: float
    beta: float
    xi: float
    eps: float

    # attributes
    x_samples: np.ndarray
    X: np.ndarray
    y: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    var: np.ndarray
    acquisition: np.ndarray

    def __init__(self, x_samples, sigma=2, noise=1e-8, length_scale=1, beta=1, xi=0.01, eps=0.01):
        self.sigma = sigma
        self.beta = beta
        self.length_scale = length_scale
        self.xi = xi
        self.x_samples = x_samples
        self.eps = eps
        self.noise = noise

    def UCB(self):
        return self.mean + self.beta * self.var

    def EI(self):
        with np.errstate(divide='warn'):
            #Z = (self.mean - np.max(self.y) - self.xi)
            Z = (self.mean - np.max(self.mean) - self.xi)
            Z_div = Z/self.var
            ei = Z * scipy.stats.norm.cdf(Z_div) + self.var * scipy.stats.norm.pdf(Z_div)
            ei[self.var <= 0.0] = 0.0
        return ei

    def PI(self):
        return scipy.stats.norm.cdf((self.mean - np.max(self.y) - self.eps)/self.var)

    def max_acquisition(self, acq_func):
        if acq_func == "UCB":
            self.acquisition = self.UCB()
        elif acq_func == "PI":
            self.acquisition = self.PI()
        else:
            self.acquisition = self.EI()
        ix = np.argmax(self.acquisition)
        return self.x_samples[ix]

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

    def posterior(self):
        K = self.rbf_fast(self.X, self.X) + self.noise**2 * np.eye(len(self.X))
        K_inv = np.linalg.pinv(K)
        K_s = self.rbf_fast(self.X, self.x_samples)
        K_ss = self.rbf_fast(self.x_samples, self.x_samples)
        self.mean = K_s.T.dot(K_inv).dot(self.y)
        self.cov = K_ss - K_s.T.dot(K_inv).dot(K_s)
        var = []
        for i in range(len(self.cov)):
            var.append(self.cov[i, i])
        self.var = np.asarray(var).reshape((len(self.cov), 1))

    def optimise(self, f, X_start, n_iter=10, acq_func="EI"):
        self.X = X_start
        print("X:\n", self.X)
        self.y = f(self.X)
        print("y:\n", self.y)
        for i in range(n_iter):
            print("=======================")
            print("iteration:      ", i)
            print("calculating posterior ...")
            self.posterior()
            print("calculating aquisition...")
            x_sample = self.max_acquisition(acq_func)
            print("current sample: ", x_sample)
            y_sample = f(x_sample)
            print("current value:  ", y_sample)
            self.X = np.vstack((self.X, x_sample))
            self.y = np.vstack((self.y, y_sample))
            print("")

        return self.X, self.y
