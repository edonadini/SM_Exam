# coding=utf-8

import numpy as np
import random as rnd
import pgutils as pg
import pgmath as pm
import algorithm as al
import os
import pandas as pd
import time


def test(root_dir, dimension, lam, r, n_iter, n_song, start_song):
    with open(os.path.join(root_dir, 'test.txt')) as f:

        test_data = f.readlines()

    test_dataset = pg.data_to_list(test_data[2:])

    with open(os.path.join(root_dir, "train.txt"), "r") as f:
        train_data = f.readlines()

    train_dataset = pg.data_to_list(train_data[2:])

    train_dataset = train_dataset[:50]
    max(train_dataset)

    song_hash = pd.read_csv(os.path.join(root_dir, "song_hash.txt"), sep="\t", header=None)

    nu = rnd.choice([0, lam])
    n_landmarks = 50
    tau = 0.5

    songs = 468
    transition_matrix = pg.transition_count(songs, train_dataset)
    #position = np.random.rand(songs, dimension)
    num_transition = np.sum(transition_matrix)
    params = pm.AlgParams(lam, nu, tau, num_transition, n_landmarks, r, dimension, n_iter)

    tic = time.perf_counter()
    X = al.single_point_algorithm(song_hash, transition_matrix, params)
    toc = time.perf_counter()
    print("total time {", toc - tic, ":0.4f} seconds")

    dummy_landmarks = [[i for i in range(songs)] for j in range(songs)]
    d2 = pm.Distances.delta(X)
    prob_matrix = np.exp(-d2) / pm.Distances.zeta(d2, X, dummy_landmarks)
    cum_sum = np.sum(prob_matrix, axis=1)
    prob_matrix = prob_matrix / cum_sum[:, np.newaxis]

    print('cum', cum_sum)
    print('sum norm', np.sum(prob_matrix, axis=1))
    print(np.sum(prob_matrix[1]))

    print(prob_matrix)
    print(np.sum(prob_matrix, axis=0))

    n_test = np.sum(pg.transition_count(songs, test_dataset[:100]))
    print(n_test)

    evaluation = pg.log_like(test_dataset[:100], prob_matrix) / n_test
    print(evaluation)

    pg.playlist_generator(n_song, start_song, song_hash, prob_matrix)
