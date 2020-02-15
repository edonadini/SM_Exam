import numpy as np
import random as rnd
from src import pgmath as pgm


def int_mapping(s):
    return list(map(int, s.split(" ")[:-1]))


def data_to_list(data):
    return list(map(int_mapping, data))


def transition_count(S, dataset):
    """

    :param S: number of songs in the database
    :param dataset: data corresponding to playlists
    :return: matrix with count of each transition

    """
    count_matrix = np.zeros((S, S))
    length = len(dataset)
    for row in range(length):
        for predecessor, successor in zip(dataset[row][:-1], dataset[row][1:]):
            count_matrix[predecessor][successor] = count_matrix[predecessor][successor] + 1
        # for j in range(len(dataset[row]) - 1):
        #    a = dataset[row][j];
        #    b = dataset[row][j + 1];
        #    count_matrix[a][b] = count_matrix[a][b] + 1
    return count_matrix


def update_landmarks(S, C, distances, params):
    if np.min(np.array([len(C[i]) / S for i in len(C)])) >= params.r:
        return

    landmarks = rnd.sample(range(S), params.n_landmarks)

    for s in range(S):
        if len(C[s]) / S < params.r:

            closest_idx = np.argmin(np.array([distances.Z[s, l] for l in landmarks]))
            closest_landmark = landmarks[closest_idx]

            if closest_landmark not in C[s]:
                C[s].append(closest_landmark)


# Mo funge, ad ogni canzone nel dataset viene associata il landmark piu vicino ed i sucessori
# osservati nelle playlist

def initialize_landmarks(S, distances, T, params):
    C = [[x for x in range(S) if T[s][x] > 0] for s in range(S)]

    # C = [[] for i in range(S)]
    # for i in range(len(T)):
    #    for j in range(len(T[i])):
    #        if T[i][j] > 0:
    #            C[i].append(j)

    update_landmarks(S, C, distances, params)

    return C


def updateU(S, T, X_old, X_new, params, dist):
    for s_p in range(S):
        dev_term = np.array([T[s_p, b] * pgm.dlU(s_p, b, s_p, dist, params) for b in range(S)])
        X_new.U[s_p] = X_old.U[s_p] + (params.tau / params.N) * (sum(dev_term) - pgm.doU(X_old, s_p, params))
    return X_new.U


def updateV(S, T, X_old, X_new, params, dist):
    for s_q in range(S):
        V_q = X_old.V[s_q]
        for s_p in range(S):
            if s_p == s_q:
                continue
            dev_term = np.array([T[s_p, b] * pgm.dlV(s_p, b, s_q, dist) for b in range(S)])
            V_q = V_q + (params.tau / params.N) * (sum(dev_term) - pgm.doV(X_old, s_p))
        X_new.V[s_q] = V_q
    return X_new.V
