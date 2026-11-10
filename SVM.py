import numpy as np
from cvxopt import matrix, solvers


def linear(x1, x2, **kwargs):
    return x1 @ x2


def rbf_kernel(x1, x2, gamma=0.1):
    sq_dists = (np.sum(x1 ** 2, axis=1, keepdims=True) + np.sum(x2 ** 2, axis=1) - 2 * np.dot(x1, x2.T))
    return np.exp(-gamma * sq_dists)


class SVM:
    def __init__(self, lower_bound=1e-6, C=1000.0, kernel_function=linear, **kernel_f_args):
        self.X = None
        self.support_vectors = None
        self.w0 = None
        self.alphas = None
        self.kernel_func = kernel_function
        self.kernel_f_args = kernel_f_args
        self.lower_bound = lower_bound
        self.C = C

    def fit(self, train_data, train_class):
        N, d = train_data.shape
        y = train_class.astype(float)
        X = train_data.astype(float)
        # Specify P, q, G, h, A, b
        P = matrix((y @ y.T) * self.kernel_func(X.T, X.T, **self.kernel_f_args))
        q = matrix(-np.ones((N, 1)))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.hstack((np.zeros(N), self.C * np.ones(N))).reshape(-1, 1))
        A = matrix(y.T)
        b = matrix(0.0)

        # Run solver
        sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()

        # Solve w and w0
        sv_c1_idx = np.where((alphas > self.lower_bound))[0][0]
        # sv_idx = np.where((alphas > LB))[0]
        K_k = self.kernel_func(X.T, X[sv_c1_idx:sv_c1_idx + 1].T, **self.kernel_f_args)
        self.w0 = y.flatten()[sv_c1_idx] - np.sum(alphas * y.flatten() * K_k.flatten())
        self.support_vectors = (alphas * y.flatten()[:, None])
        self.X = X

    def predict(self, X):
        transformed_vals = self.kernel_func(self.X.T, X.T, **self.kernel_f_args)
        scores = np.sum((self.support_vectors * transformed_vals), axis=0) + self.w0
        return np.where(scores > 0, -1, 1)
