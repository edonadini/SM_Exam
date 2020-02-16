from collections import namedtuple
from enum import Enum

from recordclass import recordclass
import numpy as np
import math

DataRepresentation = recordclass('DataRepresentation', 'U V')

AlgParams = namedtuple('AlgParams', 'lam nu tau num_transition n_landmarks r')
"""

:param lam: regularization parameter set by cross validation
:param nu: regularization parameter for dual point model
:param tau: tau predefined learning rate
:param num_transition: total number of transition of songs in the set
:param n_landmarks: number of landmarks chose in the training song set
:param r: threshold between 0 and 1, represents the amount of training data set to use

"""


def single_loss_derivative_on_entry(a, b, p, dist):
    if a != p:
        return 0
    s_term = np.array([math.exp(-dist.D[a, j] ** 2) * dist.diff[a, j, :] for j in range(len(dist.Z))])
    return 2 * (-dist.diff[a, b, :] + np.sum(s_term) / dist.Z[a])


def single_derivative_of_regularization_term_on_entry(x, p, params):
    return 2 * params.l * x[p]


def dual_loss_derivative_on_entry(a, b, p, dist):
    if a != p:
        return 0
    s_term = np.array([math.exp(-dist.D[a, j] ** 2) * dist.diff[a, j, :] for j in range(len(dist.Z))])
    return 2 * (-dist.diff[a, b, :] + np.sum(s_term) / dist.Z[a])


def dual_loss_derivative_on_exit(a, b, q, dist):
    if b != q:
        return 0
    return 2 * (dist.diff[a, b] - (math.exp(-dist.D[a, q] ** 2) * dist.diff[a, q]) / dist.Z[a])


def derivative_of_regularization_term_on_entry(self, p, params):
    return 2 * params.l * self.U[p] - 2 * params.nu * (self.V[p] - self.U[p])


def derivative_of_regularization_term_on_exit(self, p, params):
    return 2 * params.l * self.V[p] + 2 * params.nu * (self.V[p] - self.U[p])


class RepresentationType(Enum):
    SINGLE = 0
    DOUBLE = 1


class Distances:

    def __init__(self, x, representation='single'):
        if representation == 'single':
            self._type = RepresentationType.SINGLE
        else:
            self._type = RepresentationType.DOUBLE

        self.Z, self.D, self.diff = Distances.initialize(x, self._type)

    def update(self, x):
        self.Z, self.D, self.diff = Distances.initialize(x, self._type)

    @classmethod
    def initialize(cls, x, representation, landmarks):
        diff = Distances.difference_matrix(x, representation)
        d2 = Distances.delta2(diff)
        z2 = Distances.zeta2(d2, landmarks)

        return Distances(z2, d2, diff)

    @staticmethod
    def delta2(differences):

        distance_mat = [np.linalg.norm(differences[i, j, :]) for i in range(len(differences)) for j in
                        range(len(differences[i]))]

        return np.array(distance_mat).reshape(differences.shape[:2])

    @staticmethod
    def zeta2(d2, landmarks):

        z_vec = []
        for song in range(len(landmarks)):
            sum_components = np.array([d2[song, landmark] for landmark in landmarks[song]])
            z_vec.append(sum(np.exp(-(sum_components ** 2))))
        return np.array(z_vec)

    @staticmethod
    def difference_matrix(x, representation):
        if representation == RepresentationType.DOUBLE:
            y_dim = len(x.U)
            x_dim = len(x.V)

            d = len(x.U[0])

            dif_mat = np.array([(x.V[i] - x.U[j]) for i in range(x_dim) for j in range(y_dim)])
            return dif_mat.reshape((x_dim, y_dim, d))
        else:
            dim = len(x)
            d = len(x[0])

            dif_mat = np.array([(x[i] - x[j]) for i in range(dim) for j in range(dim)])
            return dif_mat.reshape((dim, dim, d))
