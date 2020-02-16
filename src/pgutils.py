import numpy as np
import random as rnd
from src import pgmath as pgm


def int_mapping(s):
    return list(map(int, s.split(" ")[:-1]))


def data_to_list(data):
    return list(map(int_mapping, data))


def transition_count(songs, dataset):
    """

    :param songs: number of songs in the database
    :param dataset: data corresponding to playlists
    :return: matrix with count of each transition

    """
    count_matrix = np.zeros((songs, songs))
    length = len(dataset)
    for row in range(length):
        for predecessor, successor in zip(dataset[row][:-1], dataset[row][1:]):
            count_matrix[predecessor][successor] = count_matrix[predecessor][successor] + 1
        # for j in range(len(dataset[row]) - 1):
        #    a = dataset[row][j];
        #    b = dataset[row][j + 1];
        #    count_matrix[a][b] = count_matrix[a][b] + 1
    return count_matrix


def update_landmarks(songs, chunk, distances, params):
    """

       :param songs: number of songs in the database
       :param chunk: candidate set of successor for each songs s_i
       :param distances: distance D, distance between two songs
       :param params: parameters defined

    """
    if np.min(np.array([len(chunk[i]) / songs for i in range(len(chunk))])) >= params.r:
        return

    landmarks = rnd.sample(range(songs), params.n_landmarks)

    for s in range(songs):
        while len(chunk[s]) / songs < params.r:
            # mi sa che devo considerare non Z ma delta perché voglio la distanza tra le canzoni
            # Z indica la partition function
            closest_idx = np.argmin(np.array([distances.D[s, j] for j in landmarks]))
            closest_landmark = landmarks[closest_idx]

            if closest_landmark not in chunk[s]:
                chunk[s].append(closest_landmark)


# Ad ogni canzone nel dataset viene associata il landmark piu vicino ed i sucessori
# osservati nelle playlist

def initialize_landmarks(songs, params, x, transition_matrix):
    chunk = [[i for i in range(songs) if transition_matrix[s][i] > 0] for s in range(songs)]

    dim = len(x)
    distance_mat = [[np.linalg.norm(x[i] - x[j]) for j in range(dim)] for i in range(dim)]
    initial_distance = np.array(distance_mat).reshape((dim, dim))

    if np.min(np.array([len(chunk[i]) / songs for i in range(len(chunk))])) >= params.r:
        return chunk

    landmarks = rnd.sample(range(songs), params.n_landmarks)

    for s in range(songs):
        while len(chunk[s]) / songs < params.r:
            # mi sa che devo considerare non Z ma delta perché voglio la distanza tra le canzoni
            # Z indica la partition function
            closest_idx = np.argmin(np.array([initial_distance[s][j] for j in landmarks]))
            closest_landmark = landmarks[closest_idx]

            if closest_landmark not in chunk[s]:
                chunk[s].append(closest_landmark)

    return chunk


def zeta(d, x, chunk):
    for a in range(len(x)):
        z_vec = [sum(np.exp(-(d[a, j] ** 2))) for j in chunk[a]]
        return np.array(z_vec)


def update_song_entry_vector(songs, transition_matrix, position_old, position_new, params, dist):
    for s in range(songs):
        dev_term = np.array([transition_matrix[s, b] *
                             pgm.loss_derivative_on_entry(s, b, s, dist) for b in range(songs)])

        position_new[s] = position_old[s] + (params.tau / params.N) * \
                          (sum(dev_term) - pgm.derivative_of_regularization_term_on_entry(position_old, s, params))
    return position_new


"""
# nel single points questo non mi serve
def update_song_exit_vector(S, T, X_old, X_new, params, dist):
    for s_q in range(S):
        V_q = X_old.V[s_q]
        for s_p in range(S):
            if s_p == s_q:
                continue
            dev_term = np.array([T[s_p, b] * pgm.dlV(s_p, b, s_q, dist) for b in range(S)])
            V_q = V_q + (params.tau / params.N) * (sum(dev_term) - pgm.doV(X_old, s_p))
        X_new.V[s_q] = V_q
    return X_new.V
"""
