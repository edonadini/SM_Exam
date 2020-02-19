# coding=utf-8

from collections import namedtuple
import numpy as np
import math as mt

AlgParams = namedtuple('AlgParams', 'lam tau num_transition n_landmarks r dimension n_iter')
"""

:param lam: regularization parameter set by cross validation
:param tau: tau predefined learning rate
:param num_transition: total number of transition of songs in the set
:param n_landmarks: number of landmarks chose in the training song set
:param r: threshold between 0 and 1, represents the amount of training data set to use

"""


class Distances:

    def __init__(self, x, chunk):
        self.Z, self.D, self.diff = Distances.initialize_aux(x, chunk)

    def update(self, x, chunk):
        self.Z, self.D, self.diff = Distances.initialize_aux(x, chunk)

    @classmethod
    def initialize_aux(cls, x, chunk):
        d = Distances.delta(x)
        z = Distances.zeta(d, x, chunk)
        diff = Distances.difference_matrix(x)
        return z, d, diff

    @staticmethod
    def delta(x):
        dim = len(x)
        distance_mat = [[np.linalg.norm(x[i] - x[j]) for j in range(dim)] for i in range(dim)]
        return np.array(distance_mat).reshape((dim, dim))

    @staticmethod
    def zeta(d, x, chunk):
        z_vec = []
        for a in range(len(x)):
            sum_terms = np.array([d[a, landmark] for landmark in chunk[a]])
            z_vec.append(np.sum(np.exp(-(sum_terms ** 2))))
        return np.array(z_vec)

    @staticmethod
    def difference_matrix(x):
        dim = len(x)
        d = len(x[0])

        dif_mat = np.array([(x[i] - x[j]) for i in range(dim) for j in range(dim)])
        return dif_mat.reshape((dim, dim, d))


def loss_derivative_on_entry(dist):
    _, _, d = dist.diff.shape
    mul_terms = np.repeat(np.exp(-np.square(dist.D))[:, :, np.newaxis], d, axis=2)
    sum_terms = np.sum(np.multiply(mul_terms, dist.diff), axis=1)
    sum_terms = np.divide(sum_terms, np.repeat(dist.Z[:, np.newaxis], d, axis=1))
    return 2 * (-dist.diff * sum_terms)


def derivative_of_regularization_term_on_entry(x, params):
    return np.array(2 * params.lam * x)
