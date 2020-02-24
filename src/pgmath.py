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
        diff = Distances.difference_matrix(x)
        d = Distances.delta(diff)
        z = Distances.zeta(d)
        return z, d, diff

    @staticmethod
    def delta(diff):
        dim, *_ = diff.shape
        distance_mat = np.array([[np.linalg.norm(diff[i,j]) if i < j else 0 for j in range(dim)] for i in range(dim)]).reshape((dim, dim))
        distance_mat = distance_mat + distance_mat.T
        return distance_mat

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

    sum_terms = np.divide(np.sum(np.multiply(np.exp(-np.square(dist.D))[:, :, np.newaxis], dist.diff), axis=1),
                          dist.Z[:, np.newaxis])
    return 2 * (-dist.diff * sum_terms)


def derivative_of_regularization_term(x, params):
    return np.array(2 * params.lam * x)
