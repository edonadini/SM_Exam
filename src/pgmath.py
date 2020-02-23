from collections import namedtuple
import numpy as np

AlgParams = namedtuple('AlgParams', 'lam tau num_transition dimension n_iter')
"""

:param lam: regularization parameter set by cross validation
:param tau: tau predefined learning rate
:param num_transition: total number of transition of songs in the set
:param n_landmarks: number of landmarks chose in the training song set
:param r: threshold between 0 and 1, represents the amount of training data set to use

"""


class Distances:

    def __init__(self, x):
        self.Z, self.D, self.diff = Distances.initialize_aux(x)

    def update(self, x):
        self.Z, self.D, self.diff = Distances.initialize_aux(x)

    @classmethod
    def initialize_aux(cls, x):
        d = Distances.delta(x)
        z = Distances.zeta(d)
        diff = Distances.difference_matrix(x)
        return z, d, diff

    @staticmethod
    def delta(x):
        dim = len(x)
        distance_mat = [[np.linalg.norm(x[i] - x[j]) for j in range(dim)] for i in range(dim)]
        return np.array(distance_mat).reshape((dim, dim))

    @staticmethod
    def zeta(d):
        return np.sum(np.exp(-np.square(d)), axis=1)

    @staticmethod
    def difference_matrix(x):
        dim, d = x.shape

        dif_mat = np.array([(x[i] - x[j]) for i in range(dim) for j in range(dim)])
        return dif_mat.reshape((dim, dim, d))


def loss_derivative(dist):
    _, _, d = dist.diff.shape
    mul_terms = np.repeat(np.exp(-np.square(dist.D))[:, :, np.newaxis], d, axis=2)
    sum_terms = np.sum(np.multiply(mul_terms, dist.diff), axis=1)
    sum_terms = np.divide(sum_terms, np.repeat(dist.Z[:, np.newaxis], d, axis=1))
    return 2 * (-dist.diff * sum_terms)


def derivative_of_regularization_term(x, params):
    return np.array(2 * params.lam * x)


