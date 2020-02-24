import numpy as np
import random as rnd
import math as mt
import pgmath as pgm


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


def update_songs( position_new, transition_matrix, position_old, params, dist):
    _, d = position_old.shape

    mul_term = (params.tau / params.num_transition)

    position_new[:, :] = position_old + mul_term * np.subtract(
                                np.sum(np.multiply(transition_matrix[:, :, np.newaxis], pgm.loss_derivative(dist)), axis=1),
                                pgm.derivative_of_regularization_term(position_old, params))


def log_like(test_set, probability_matrix):
    value = 0
    for playlist in range(len(test_set)):
        for predecessor, successor in zip(test_set[playlist][:-1], test_set[playlist][1:]):
            value = value + mt.log(probability_matrix[predecessor, successor])

    return value


def playlist_generator(num_song, init_song, prob_matrix, song_hash):

    # Start of the time
    t = 0
    last_song = init_song
    # The music playlist
    playlist = [last_song]
    # initial probability
    prob = 1
    # while loop - the Markov chain did not reach 5 we go on
    while t < num_song:
        t = t + 1
        # List of played songs
        playlist.append(
            np.random.choice(np.array(song_hash.iloc[0:2200][0]), replace=True, p=prob_matrix[last_song]))
        prob = prob * prob_matrix[last_song, playlist[-1]]
        last_song = playlist[-1]

    titles = [song_hash.iloc[i][1] for i in playlist]
    print("Song in the playlist: ", titles)
    print("Probability of the possible sequence of states: ", str(prob))
