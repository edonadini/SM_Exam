# coding=utf-8

import numpy as np
import random as rnd
import math as mt
import pgmath as pgm
import pandas as pd
import  os

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
        if len(chunk[s]) / songs < params.r:
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
        if len(chunk[s]) / songs < params.r:
            closest_idx = np.argmin(np.array([initial_distance[s][j] for j in landmarks]))
            closest_landmark = landmarks[closest_idx]

            if closest_landmark not in chunk[s]:
                chunk[s].append(closest_landmark)

    return chunk


def update_song_entry_vector(transition_matrix, position_old, params, dist):
    _, d = position_old.shape
    regularization_derivative = pgm.derivative_of_regularization_term_on_entry(position_old, params)
    mul_term = (params.tau / params.num_transition)

    loss_derivatives = pgm.loss_derivative_on_entry(dist)

    sum_dev_term = np.multiply(np.repeat(transition_matrix[:, :, np.newaxis], d, axis=2), loss_derivatives)
    sum_dev_term = np.sum(sum_dev_term, axis=1)

    position_new = position_old + mul_term * np.subtract(sum_dev_term, regularization_derivative)

    return position_new


def log_like(test_set, probability_matrix):
    count = 0
    for i in range(len(test_set)):
        for predecessor, successor in zip(test_set[i][:-1], test_set[i][1:]):
            count = count + mt.log(probability_matrix[predecessor, successor])

    return count


def playlist_generator(num_song, current_song, num_can, prob_matrix):
    #prob_matrix = np.genfromtxt(r'C:\Users\eleon\Desktop\SM_Exam\src\latent_representation2200.csv', delimiter=',')
    root_dir = r'C:\Users\eleon\PycharmProjects\SM_Exam\data\yes_small'
    song_hash = pd.read_csv(os.path.join(root_dir, "song_hash.txt"), sep="\t", header=None)

    # Start of the time
    t = 0
    # The music playlist
    playlist = [current_song]
    # initial probability
    prob = 1
    # while loop - the Markov chain did not reach 5 we go on
    while t < num_song:
        # Incrementation of time
        t = t + 1
        # List of played songs
        playlist.append(np.random.choice(np.array(song_hash.iloc[0:num_can][0]), replace=True, p=prob_matrix[current_song]))
        prob = prob * prob_matrix[current_song, playlist[-1]]
        current_song = playlist[-1]

    # Printing the path of the Markov chain
    # Printing the number of steps
    # print(len(X)-1)
    titles = [song_hash.iloc[i][1] for i in playlist]
    print("Song in the playlist: ", titles)
    print("End state after ", num_song, " days: ", song_hash.iloc[current_song][1])
    print("Probability of the possible sequence of states: ", str(prob))
