from collections import namedtuple
import numpy as np
import numpy.ma as ma

AlgParams = namedtuple('AlgParams', 'lam tau num_transition dimension n_iter r n_landmarks')
"""

:param lam: regularization parameter set by cross validation
:param tau: tau predefined learning rate
:param num_transition: total number of transition of songs in the set
:param n_landmarks: number of landmarks chose in the training song set
:param r: threshold between 0 and 1, represents the amount of training data set to use

"""


class Distances:

    def __init__(self, x, chunks):
        self.Z, self.D, self.diff = Distances.initialize_aux(x, chunks)

    def update(self, x, chunks):
        self.Z, self.D, self.diff = Distances.initialize_aux(x, chunks)

    @classmethod
    def initialize_aux(cls, x, chunks):
        diff = Distances.difference_matrix(x, chunks)
        d = Distances.delta(diff, chunks)
        z = Distances.zeta(d)
        return z, d, diff

    @staticmethod
    def delta(diff, chunks):
        dim, *_ = diff.shape

        return np.linalg.norm(diff, axis=2)
        #distance_mat = np.array(
        #    [[np.linalg.norm(diff[i, j]) if j in chunks[j] else 0 for j in range(dim)] for i in range(dim)]).reshape(
        #    (dim, dim))
        # distance_mat = distance_mat + distance_mat.T
        #return distance_mat

    @staticmethod
    def zeta(d):
        return np.sum(np.exp(-np.square(d)), axis=1)

    @staticmethod
    def difference_matrix(x, chunks):
        dim, d = x.shape

        zero = np.zeros(d)
        dif_mat = np.array(
            [(x[i] - x[j]) if chunks[i, j, 0] == 0 else zero for i in range(dim) for j in range(dim)]).reshape(
            (dim, dim, d))

        return ma.masked_array(dif_mat, mask=chunks)


def loss_derivative(dist):
    sum_terms = np.divide(np.sum(np.multiply(np.exp(-np.square(dist.D))[:, :, np.newaxis], dist.diff), axis=1),
                          dist.Z[:, np.newaxis])
    return 2 * (-dist.diff * sum_terms)


def derivative_of_regularization_term(x, params):
    return np.array(2 * params.lam * x)
