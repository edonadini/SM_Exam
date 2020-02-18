import numpy as np
import random as rnd
import pgutils as pg
import pgmath as pm
import algorithm as al
import os
import pandas as pd

def test(root_dir, dimension, lam, r, n_iter, n_song, start_song):

    #root_dir = r'C:\Users\eleon\PycharmProjects\SM_Exam\data\yes_small'
    # root_dir = './data/yes_small'


    with open(os.path.join(root_dir, 'test.txt')) as f:
        # with open(root_dir + "\\test.txt", "r") as f:

        test_data = f.readlines()

    test_dataset = pg.data_to_list(test_data[2:])


    with open(os.path.join(root_dir, "train.txt"), "r") as f:
        train_data = f.readlines()

    train_dataset = pg.data_to_list(train_data[2:])

    song_hash = pd.read_csv(os.path.join(root_dir, "song_hash.txt"), sep="\t", header=None)


    # setting parameters
    #dimension = 2  # rnd.choice([2, 5,10,25,50,100])
    # regularization parameter set by cross validation
    #lam = 1  # rnd.choice([0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 500, 1000])
    # regularization parameter for dual point model
    nu = rnd.choice([0, lam])
    # threshold for landmark dimension
    #r = 0.2
    # number of landmark
    n_landmarks = 3 #50
    # num iteration 100 o 200
    #n_iter = 100  # rnd.choice([100, 200])
    # tau predefined learning rate
    tau = 0.5


    toy_dataset=[[0,6,3,2,1,4,2],[2,0,6,5,6,3,0],[0,5,2,1,6],[5,6,2],[6,5,2,3,5,6,2]]

    songs = 7 #len(song_hash)
    transition_matrix = pg.transition_count(songs, toy_dataset)
    position = np.random.rand(songs, dimension)
    num_transition = np.sum(transition_matrix)
    params = pg.AlgParams(lam, nu, tau, num_transition, n_landmarks, r, dimension, n_iter)

    X = al.single_point_algorithm(song_hash, transition_matrix, params)


    dummy_landmarks = [[i for i in range(songs)] for j in range(songs)]
    d2 = pm.Distances.delta(X)
    prob_matrix = np.exp(-d2) / pm.Distances.zeta(d2, X, dummy_landmarks)
    cum_sum = np.sum(prob_matrix, axis=1)
    prob_matrix = prob_matrix / cum_sum[:, np.newaxis]

    toy_test=[[0,4,5,3,6],[4,3,2,6],[0,2,1,0,5]]
    # evaluate test performance using the average log-likelihood as metric
    # it is defined as log(Pr(d_test))/n_test)
    # n_test number of transition in the test set
    # songs = 7 #len(song_hash)
    n_test = np.sum(pg.transition_count(songs, toy_test)) #test_dataset
    print(n_test)

    evaluation = pg.log_like(toy_test, prob_matrix) / n_test
    print(evaluation)
    # result = sum( sum(dimension.D[s, i.index()] ) for i in range(len(test_dataset)))


    # generate a playlist
    pg.playlist_generator(n_song , start_song , song_hash, prob_matrix)

