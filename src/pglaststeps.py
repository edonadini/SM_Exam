# coding=utf-8

import numpy as np
import pgutils as pg
import pgmath as pm
import algorithm as al
import os
import time
from numpy import savetxt


def latent_representation(root_dir, dimension, lam, r, n_iter, tau):
    with open(os.path.join(root_dir, "train.txt"), "r") as f:
        train_data = f.readlines()
    train_dataset = pg.data_to_list(train_data[2:])

    songs = 3168

    max(train_dataset)

    n_landmarks = 50

    jump_matrix = pg.transition_count(songs, train_dataset)
    num_transition = np.sum(jump_matrix)

    params = pm.AlgParams(lam, tau, num_transition, n_landmarks, r, dimension, n_iter)

    tic = time.perf_counter()
    X = al.single_point_algorithm(songs, jump_matrix, params)
    toc = time.perf_counter()
    savetxt('latent_representation_50.csv', X, delimiter=' ')
    print("total time {", toc - tic, ":0.4f} seconds")


def tran_matrix(songs, x):
    dummy_landmarks = [[i for i in range(songs)] for j in range(songs)]
    d2 = pm.Distances.delta(x)
    prob_matrix = np.exp(-d2) / pm.Distances.zeta(d2, x, dummy_landmarks)[:, np.newaxis]
    cum_sum = np.sum(prob_matrix, axis=1)
    prob_matrix = prob_matrix / cum_sum[:, np.newaxis]

    savetxt('prob_matrix_50.csv', prob_matrix, delimiter=' ')


def evaluation_loss(root_dir, songs, prob_matrix):
    with open(os.path.join(root_dir, 'test.txt')) as f:
        # with open(root_dir + "\\test.txt", "r") as f:

        test_data = f.readlines()

    test_dataset = pg.data_to_list(test_data[2:])
    n_test = np.sum(pg.transition_count(songs, test_dataset[:100]))

    evaluation = pg.log_like(test_dataset[:100], prob_matrix) / n_test
    print('loss on test set ', evaluation)
