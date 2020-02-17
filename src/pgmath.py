from collections import namedtuple

from recordclass import recordclass
import numpy as np
import math as mt

# DataRepresentation = recordclass('DataRepresentation', 'U V')
AlgParams = namedtuple('AlgParams', 'lam nu tau num_transition n_landmarks r dimension n_iter')
"""

:param lam: regularization parameter set by cross validation
:param nu: regularization parameter for dual point model
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


def loss_derivative_on_entry(a, b, p, dist):
    if a != p:
        return 0
    s_term = np.array([mt.math.exp(-dist.D[a, j] ** 2) * dist.diff[a, j, :] for j in range(len(dist.Z))])
    return 2 * (-dist.diff[a, b, :] + np.sum(s_term) / dist.Z[a])


def derivative_of_regularization_term_on_entry(x, p, params):
    return 2 * params.lam * x[p]


def log_like(test_set, probability_matrix):
    count = 0
    for i in range(len(test_set)):
        for predecessor, successor in zip(test_set[i][:-1], test_set[i][1:]):
            count = count + mt.log(probability_matrix[predecessor, successor])

    return count
"""
    dual point representation
    class Distances:

    def __init__(self, x):
        self.Z, self.D, self.diff = Distances.initialize(x)

    def update(self, x):
        self.Z, self.D, self.diff = Distances.initialize(x)
    
    @classmethod
    def initialize(cls, self):
        d2 = Distances.delta2(self)
        diff = Distances.difference_matrix(self)
        z2 = Distances.zeta2(d2)

        return Distances(z2, d2, diff)
        
    @staticmethod
    def delta2(self):
        x_dim = len(self.V)
        y_dim = len(self.U)

        distance_mat = [[np.linalg.norm(self.V[i] - self.U[j]) for j in range(x_dim)] for i in range(y_dim)]

        return np.array(distance_mat).reshape((x_dim, y_dim))

    @staticmethod
#come faccio a passargli il landmark?
    def zeta2(d2, chunk):
        z2_vec = [sum(np.exp(-(d2[:, idx] ** 2))) for idx in range(len(chunk))]
        return np.array(z2_vec)

    @staticmethod
    def difference_matrix(self):
        y_dim = len(self.U)
        x_dim = len(self.V)

        d = len(self.U[0])

        dif_mat = np.array([(self.V[i] - self.U[j]) for i in range(x_dim) for j in range(y_dim)])
        return dif_mat.reshape((x_dim, y_dim, d))


def loss_derivative_on_entry(a, b, p, dist):
    if a != p:
        return 0
    s_term = np.array([math.exp(-dist.D[a, j] ** 2) * dist.diff[a, j, :] for j in range(len(dist.Z))])
    return 2 * (-dist.diff[a, b, :] + np.sum(s_term) / dist.Z[a])


def loss_derivative_on_exit(a, b, q, dist):
    if b != q:
        return 0
    return 2 * (dist.diff[a, b] - (math.exp(-dist.D[a, q] ** 2) * dist.diff[a, q]) / dist.Z[a])


def derivative_of_regularization_term_on_entry(self, p, params):
    return 2 * params.l * self.U[p] - 2 * params.nu * (self.V[p] - self.U[p])


def derivative_of_regularization_term_on_exit(self, p, params):
    return 2 * params.l * self.V[p] + 2 * params.nu * (self.V[p] - self.U[p])
"""
