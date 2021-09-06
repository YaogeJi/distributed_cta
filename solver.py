import numpy as np
from projection import euclidean_proj_l1ball as proj


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
        r = np.linalg.norm(ground_truth, ord=1) * 1.01
        loss = []
        N, d = X.shape
        # initialize iterates
        theta = 0.5 * np.ones((d, 1))
        # calculate value we need to use.
        C_1 = X.T @ X
        C_2 = X.T @ Y
        a, _ = np.linalg.eig(C_1)
        max_eig = np.max(a)

        # define gradient methods
        def _lagrangian(t):
            t = t - self.gamma * (1 / N * (C_1 @ t - C_2))
            temp = np.sign(t) * np.clip(np.abs(t) - self.gamma * self.lmda, 0, None)
            if not self.projecting or np.linalg.norm(temp, ord=1) <= r:
                t = temp
            else:
                t = proj(t, r)
            return t

        def _projected(t):
            # r = np.linalg.norm(ground_truth, ord=1)
            t = t - self.gamma * (1 / N * (C_1 @ t - C_2))
            t = proj(t, r)
            return t

        # iterates!
        log_loss = []
        for step in range(self.max_iteration):
            theta_last = theta.copy()
            if self.iter_type == "lagrangian":
                theta = _lagrangian(theta)
            elif self.iter_type == "projected":
                theta = _projected(theta)
            if ground_truth is not None:
                log_loss.append(np.log(np.linalg.norm(theta - ground_truth, ord=2) ** 2))
            if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition:
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2)  < self.terminate_condition:
                print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
                return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, log_loss

    def show_param(self):
        return [self.gamma, self.constraint_param]


class DistributedLasso(Lasso):
    def __init__(self, max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting, w):
        super(DistributedLasso, self).__init__(max_iteration, gamma, terminate_condition, iter_type, constraint_param, projecting)
        self.w = w
        self.m = self.w.shape[0]

    def fit(self, X, Y, ground_truth, verbose):
        # Initialize parameters we need
        r = np.linalg.norm(ground_truth, ord=1) * 1.01
        N, d = X.shape
        n = int(N / self.m)
        # Initialize iterates
        theta = 0.5 * np.ones((self.m, d))
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
        max_eig = 0
        for sth in D:
            a, _ = np.linalg.eig(sth)
            if np.max(a) > max_eig:
                max_eig = np.max(a)
        print(max_eig)
        beta = N * self.gamma / (max_eig * self.gamma + n)
        # print(N/max_eig)
        # define gradient methods

        def _lagrangian(t):
            con = self.w @ t
            for i in range(self.m):
                t[i] = (con[i] - self.gamma / n * (-E[i] + t[i].T @ D[i])) * beta / (self.m * self.gamma) + t[
                    i] * (self.m * self.gamma - beta) / (self.m * self.gamma)
                # temp = np.sign(t[i]) * np.clip(np.abs(t[i]) - self.gamma * self.lmda, 0, None)
                temp = np.sign(t[i]) * np.clip(np.abs(t[i]) - beta * self.lmda / self.m, 0, None)
                if not self.projecting or np.linalg.norm(temp, ord=1) <= r:
                    t[i] = temp
                else:
                    # projection
                    temp = np.expand_dims(t[i].copy(), axis=1)
                    temp = proj(temp, r)
                    t[i] = temp.squeeze()
            return t

        def _projected(t):
            r = np.linalg.norm(ground_truth, ord=1)
            for i in range(self.m):
                t[i] = (t[i] - self.gamma / n * (-E[i] + t[i].T @ D[i]))
            con = self.w @ t
            for i in range(self.m):
                t[i] = con[i]
                temp = np.expand_dims(t[i].copy(), axis=1)
                temp = proj(temp, r)
                t[i] = temp.squeeze()
            return t
        # iterates!

        log_loss = []
        for step in range(self.max_iteration):
            if verbose:
                if step % 100 == 0 and step != 0:
                    if ground_truth is not None:
                        print("{}/{}, log loss = {}".format(step, self.max_iteration, log_loss[-1]))
                    else:
                        print("{}/{}".format(step, self.max_iteration))
            if ground_truth is not None:
                "optimization error has bug here. shape of comparison is not consensus."
                log_loss.append(np.log(np.linalg.norm(theta - np.repeat(ground_truth.T, self.m, axis=0), ord=2) ** 2 / self.m))
            theta_last = theta.copy()
            if self.iter_type == "lagrangian":
                theta = _lagrangian(theta)
            elif self.iter_type == "projected":
                theta = _projected(theta)
            else:
                raise NotImplementedError

            if np.linalg.norm(theta - theta_last, ord=2) < self.terminate_condition * np.sqrt(self.m):
            # if np.linalg.norm(theta - theta_last, ord=2) / np.linalg.norm(theta_last, ord=2) < self.terminate_condition:
                print("Early convergence at step {} with log loss {}, I quit.".format(step, log_loss[-1]))
                return theta, log_loss
        else:
            print("Max iteration, I quit.")
            return theta, log_loss