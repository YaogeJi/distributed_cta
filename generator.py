import numpy as np
from sklearn import linear_model
from solver import Lasso


class Generator:
    def __init__(self, N, d, s, k, sigma):
        self.N = N
        self.d = d
        self.s = s
        self.k = k
        self.sigma = sigma

    def generate(self, solver=False):
        # generating sample
        z = np.random.normal(0, 1, (self.N, self.d)) / np.sqrt(1-self.k**2)
        X = np.ones((self.N, self.d))
        X[:, 0] = z[:, 0]
        for i in range(1, self.d):
            X[:, i] = self.k * X[:, i - 1] + z[:, i]

        # noise
        epsilon = np.random.normal(0, self.sigma ** 2, (int(self.N), 1))

        # generating ground truth
        # theta = np.zeros((self.d, 1))
        # theta[0:self.s] = 1
        # Y = X @ theta + epsilon
        # return X, Y, theta
        theta = np.random.normal(0, 1, (self.d, 1))
        theta_abs = np.abs(theta)
        threshold = np.quantile(theta_abs, 1 - self.s / self.d)
        mask = theta_abs > threshold
        theta = mask * theta
        Y = X @ theta + epsilon
        if solver:
            clf = linear_model.LassoCV(max_iter=1e5, cv=10)
            clf.fit(X, Y)
            optimal_lambda = clf.alpha_
            theta_hat = clf.coef_
            min_stat_error = np.linalg.norm(theta_hat - np.squeeze(theta), ord=2) ** 2
            return X, Y, theta, optimal_lambda, min_stat_error
        else:
            return X, Y, theta, False, False
