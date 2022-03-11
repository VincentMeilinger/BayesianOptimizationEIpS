import scipy
import torch
from torch import tensor
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound

class BOTorch:

    # parameters
    sigma: float
    noise: float
    length_scale: float


    # attributes
    bounds: tensor
    resolutions: tensor

    X: tensor
    X_samples: tensor
    y_samples: tensor
    acquisition: tensor

    def __init__(self, sigma=2, noise=1e-8, length_scale=1):
        self.sigma = sigma
        self.length_scale = length_scale
        self.noise = noise

    def optimize(self, f, bounds: tensor, X_samples: tensor, aq_func="EI", n_iter=10):
        self.X_samples = X_samples
        print("X_samples:\n", self.X_samples)

        self.y_samples = f(X_samples)
        print("y_samples:\n", self.y_samples)

        for i in range(n_iter):
            print("=======================")
            print("iteration:      ", i)
            print("_______________________")
            gp = SingleTaskGP(self.X_samples, self.y_samples)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)

            UCB = UpperConfidenceBound(gp, beta=0.1)
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
            print("next sample: ", candidate)

            y_new_sample = f(candidate)
            print("current value:  ", y_new_sample)

            self.X_samples = torch.vstack((self.X_samples, candidate))
            self.y_samples = torch.vstack((self.y_samples, y_new_sample))
            print("")

        return self.X_samples, self.y_samples