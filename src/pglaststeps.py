# coding=utf-8

import numpy as np
import pgutils as pg
import pgmath as pm
import algorithm as al
import os
import time
from numpy import savetxt


def latent_representation(root_dir, dimension, lam, n_iter, tau):
    with open(os.path.join(root_dir, "train.txt"), "r") as f:
        train_data = f.readlines()
    train_dataset = pg.data_to_list(train_data[2:])

    train_dataset = train_dataset[:400]
    songs = 2200

    jump_matrix = pg.transition_count(songs, train_dataset)
    num_transition = np.sum(jump_matrix)

    params = pm.AlgParams(lam, tau, num_transition, dimension, n_iter)

    tic = time.perf_counter()
    X = al.single_point_algorithm(songs, jump_matrix, params)
    toc = time.perf_counter()

    savetxt('latent_representation2200.csv', X, delimiter=' ')
    print("total time {", toc - tic, ":0.4f} seconds")


def tran_matrix(file_path):
    x = np.genfromtxt(file_path, delimiter=' ')
    d2 = pm.Distances.delta(x)
    prob_matrix = np.exp(-d2) / pm.Distances.zeta(d2)
    cum_sum = np.sum(prob_matrix, axis=1)
    prob_matrix = prob_matrix / cum_sum[:, np.newaxis]

    output_file = os.path.join(os.path.dirname(file_path), 'prob_matrix_2200.csv')
    savetxt(output_file, prob_matrix, delimiter=',')


def evaluation_loss(root_dir, songs, prob_matrix):
    with open(os.path.join(root_dir, 'test.txt')) as f:
        test_data = f.readlines()

    test_dataset = pg.data_to_list(test_data[2:])
    n_test = np.sum(pg.transition_count(songs, test_dataset[:100]))

    evaluation = pg.log_like(test_dataset[:100], prob_matrix) / n_test
    print('loss on test set ', evaluation)
