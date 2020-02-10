from recordclass import recordclass
import numpy as np
import math

DataRepresentation = recordclass('DataRepresentation', 'U V')
Distances = recordclass('DistanceMatrices', 'Z D diff')
AlgParams = recordclass('AlgParams', 'l nu tau N')


def delta2(X):
    x_dim = len(X.U)
    y_dim = len(X.V)

    distance_mat = [[np.linalg.norm(X.U[i] - X.V[j]) for j in range(x_dim)] for i in range(y_dim)]

    return np.array(distance_mat).reshape((x_dim, y_dim))


def zeta2(distance_mat):
    z2_vec = [sum(np.exp(-(distance_mat[idx, :] ** 2))) for idx in len(distance_mat)]
    return np.array(z2_vec)


def difference_matrix(X):
    y_dim = len(X.U)
    x_dim = len(X.V)

    dif_mat = np.array([[X.V[i] - X.U[j] for j in y_dim] for i in x_dim])
    return dif_mat.reshape((x_dim, y_dim))


def dlU(a, b, p, dist):
    if a != p:
        return 0
    s_term = np.array([math.exp(-dist.D[a, l] ** 2) * dist.diff[a, l, :] for l in range(len(dist.Z))])
    return 2 * (-dist.diff[a, b, :] + np.sum(s_term) / dist.Z[a])


def dlV(a, b, q, dist):
    if b != q:
        return 0
    return 2 * (dist.diff[a, b] - (math.exp(-dist.D[a, q] ** 2) * dist.diff[a, q]) / dist.Z[a])


def doU(X, p, params):
    return 2 * params.l * X.U[p] - 2 * params.nu * (X.V[p] - X.U[p])


def doV(X, p, params):
    return 2 * params.l * X.V[p] + 2 * params.nu * (X.V[p] - X.U[p])
