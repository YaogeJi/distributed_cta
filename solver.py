import numpy as np
import time
from projection import proj_l1ball as proj
from sklearn import linear_model
import scipy

class Solver:
    def __init__(self, max_iteration, gamma, terminate_condition):
        self.max_iteration = int(max_iteration)
        self.gamma = gamma
        self.terminate_condition = terminate_condition

    def fit(self, X, Y, ground_truth, verbose):
        raise NotImplementedError

    def show_param(self):
        raise NotImplementedError


class Lasso(Solver):
    def __init__(self, max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting):
        super(Lasso, self).__init__(max_iteration, gamma, terminate_condition)
        self.iter_type = iter_type
        self.constraint_param = constraint_param
        if iter_type == "lagrangian":
            self.lmda = constraint_param
        elif iter_type == "projected":
            self.r = constraint_param
        self.projecting = projecting

    def fit(self, X, Y, ground_truth, verbose):
        # initialize parameters we need
        r = np.linalg.norm(ground_truth, ord=1)
        loss = []
        N, d = X.shape
        # initialize iterates
        theta = 0.0 * np.ones((1, d))
        # calculate value we need to use.
        # C_1 = X.T @ X
        # C_2 = X.T @ Y
        #  a, _ = np.linalg.eig(C_1)
        # max_eig = np.max(a)

        # define gradient methods
        # def _lagrangian(t):
        #     t = t - self.gamma * (1 / N * (C_1 @ t - C_2))
        #     temp = np.sign(t) * np.clip(np.abs(t) - self.gamma * self.lmda, 0, None)
        #     if not self.projecting or np.linalg.norm(temp, ord=1) <= r:
        #         t = temp
        #     else:
        #         raise ValueError
        #         t = proj(t, r)
        #     return t

        def _projected(t):
            r = np.linalg.norm(ground_truth, ord=1)
            t = t - self.gamma / N * (t @ X.T - Y.T) @ X
            t = proj(t, r)
            return t

        # iterates!
        loss_matrix = []
        for step in range(self.max_iteration):
            theta_last = theta.copy()
            if self.iter_type == "lagrangian":
                theta = _lagrangian(theta)
            elif self.iter_type == "projected":
                theta = _projected(theta)
            if ground_truth is not None:
                loss_matrix.append(np.linalg.norm(theta - ground_truth.T, ord=2) ** 2)
                if step % 100 == 0:
                    print(step, loss_matrix[-1])
            # if np.linalg.norm(theta-theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2)  < self.terminate_condition:
            #     print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
            #     return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, loss_matrix

    def show_param(self):
        return [self.gamma, self.constraint_param]


class DistributedLasso(Lasso):
    def __init__(self, max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting, w):
        super(DistributedLasso, self).__init__(max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting)
        self.w = w
        self.m = self.w.shape[0]

    def fit(self, X, Y, ground_truth, verbose):
        # Initialize parameters we need
        r = np.linalg.norm(ground_truth, ord=1)
        N, d = X.shape
        assert N % self.m == 0, "sample size {} is indivisible by {} nodes.".format(N, self.m)
        n = int(N / self.m)
        # Initialize iterates
        theta = 0.0 * np.ones((self.m, d, 1))
        x = X.reshape(self.m, n, d)
        y = Y.reshape(self.m, n, 1)
        # D = x.transpose(0,2,1) @ x
        # E = x.transpose(0,2,1) @ y
        def _lagrangian(t):
            raise NotImplementedError("not implemented lagrangian atc yet, check for distributed_optimization_atc(there exists error: should not contain projection)")

        def _projected(t):
            r = np.linalg.norm(ground_truth, ord=1)
            t = self.w @ t.squeeze(axis=2)
            t = np.expand_dims(t, axis=2)
            t = t - self.gamma / n * x.transpose(0,2,1) @ (x @ t - y)
            t = t.squeeze(axis=2)
            t = (proj(t, r)).reshape(self.m,d,1)
            return t
        # iterates!

        loss_matrix = []
        for step in range(self.max_iteration):
            if verbose:
                if step % 100 == 0 and step != 0:
                    if ground_truth is not None:
                        print("{}/{}, loss = {}".format(step, self.max_iteration, loss_matrix[-1]))
                    else:
                        print("{}/{}".format(step, self.max_iteration))
            if ground_truth is not None:
                "optimization error has bug here. shape of comparison is not consensus."
                loss_matrix.append(np.linalg.norm(theta.squeeze() - np.repeat(ground_truth.T, self.m, axis=0), ord=2) ** 2 / self.m)
            theta_last = theta.copy()
            if self.iter_type == "lagrangian":
                theta = _lagrangian(theta)
            elif self.iter_type == "projected":
                theta = _projected(theta)
            else:
                raise NotImplementedError

            # if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2) < self.terminate_condition:

            # if np.max(np.linalg.norm(theta-theta_last, ord=2, axis=1)) < self.terminate_condition:
            #     print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
            #     return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, loss_matrix


class LocalizedLasso(Lasso):
    def __init__(self, max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting, m):
        super(LocalizedLasso, self).__init__(max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting)
        self.m = m

    def fit(self, X, Y, ground_truth, verbose):
        # Initialize parameters we need
        r = np.linalg.norm(ground_truth, ord=1)
        N, d = X.shape
        assert N % self.m == 0, "sample size {} is indivisible by {} nodes.".format(N, self.m)
        n = int(N / self.m)
        # Initialize iterates
        theta = 0.0 * np.ones((self.m, d))
        # Block data
        x = []
        y = []
        for i in range(self.m):
            x.append(X[n * i:n * (i + 1), :])
            y.append(Y[n * i:n * (i + 1), :])
            D = []
            E = []
        # block value we need to use
        for i in range(self.m):
            D.append(x[i].T @ x[i])
            E.append(y[i].T @ x[i])
        # max_eig = 0
        # for sth in D:
        #     a, _ = np.linalg.eig(sth)
        #    if np.max(a) > max_eig:
        #        max_eig = np.max(a)
        # print("max_eig of X.T @ X")
        # print(max_eig)
        # beta = N * self.gamma / (max_eig * self.gamma + n)
        # print(N/max_eig)
        # define gradient methods

        def _lagrangian(t):
            raise NotImplementedError("not implemented lagrangian atc yet, check for distributed_optimization_atc(there exists error: should not contain projection)")

        def _projected(t):
            r = np.linalg.norm(ground_truth, ord=1)
            for i in range(self.m):
                t[i] = (t[i] - self.gamma / n * (-E[i] + t[i].T @ D[i]))
            for i in range(self.m):
                # t[i] = con[i]
                # print(t[i].shape)
                # print(t[i].shape)
                # print(r)
                # print(np.linalg.norm(t[i], ord=1))
                t[i] = proj(t[i], r)
                # print(np.linalg.norm(t[i], ord=1))
                # print("-------")
            return t
        # iterates!

        loss_matrix = []
        for step in range(self.max_iteration):
            if verbose:
                if step % 100 == 0 and step != 0:
                    if ground_truth is not None:
                        print("{}/{}, loss = {}".format(step, self.max_iteration, loss_matrix[-1]))
                    else:
                        print("{}/{}".format(step, self.max_iteration))
            if ground_truth is not None:
                "optimization error has bug here. shape of comparison is not consensus."
                loss_matrix.append(np.linalg.norm(theta - np.repeat(ground_truth.T, self.m, axis=0), ord=2) ** 2 / self.m)
            theta_last = theta.copy()
            if self.iter_type == "lagrangian":
                theta = _lagrangian(theta)
            elif self.iter_type == "projected":
                theta = _projected(theta)
            else:
                raise NotImplementedError

            # if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2) < self.terminate_condition:

            # if np.max(np.linalg.norm(theta-theta_last, ord=2, axis=1)) < self.terminate_condition:
            #     print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
            #     return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, loss_matrix


class SolverLasso(Lasso):
    def fit(self, X, Y, ground_truth, verbose):
        clf = linear_model.Lasso(alpha=self.lmda, fit_intercept=False, max_iter=self.max_iteration)
        clf.fit(X, Y)
        theta = clf.coef_
        loss_matrix = []
        if ground_truth is not None:
            loss_matrix.append(np.linalg.norm(theta - ground_truth.squeeze(), ord=2) ** 2)
        return theta, loss_matrix


class SolverDistributedLasso(DistributedLasso):
    def fit(self, X, Y, ground_truth, verbose):
        tuta_timer = time.time()
        N, d = X.shape
        n = int(N / self.m)
        x, y = [], []
        for i in range(self.m):
            x.append(X[n*i:n*(i+1), :])
            y.append(Y[n*i:n*(i+1), :])
        X_2 = scipy.linalg.block_diag(*x)
        kron_L = np.kron(np.eye(self.m) - self.w, np.eye(d))

        _, s, VT = scipy.linalg.svd(kron_L)
        SIGMA = np.diag(np.sqrt(s))
        tmp = np.sqrt(N / self.gamma) * SIGMA @ VT

        X_tuta = np.concatenate((X_2, tmp), axis=0)
        Y_tuta = np.concatenate((Y, np.zeros((self.m*d, 1))), axis=0)
        tuta_finish = time.time()
        print("calculating kron matrix takes {} seconds".format(tuta_finish - tuta_timer))
        clf = linear_model.Lasso(alpha=self.lmda / self.m * N / (N + self.m * d), fit_intercept=False, max_iter=self.max_iteration)
        clf.fit(X_tuta, Y_tuta)
        theta = clf.coef_
        loss_matrix = []
        if ground_truth is not None:
            loss_matrix.append(np.linalg.norm(theta - np.tile(np.squeeze(ground_truth), self.m), ord=2) ** 2 / self.m)
        return theta, loss_matrix
