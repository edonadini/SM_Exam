# coding=utf-8
import time

import numpy as np
import pgmath as pm
import pgutils as pu


def single_point_algorithm(songs, transition_matrix, params):
    # assignment a random position at each song in the target space
    position = np.random.rand(songs, params.dimension)
    position_new = np.empty_like(position)

    # error for stopping criteria mean square error
    squared_error = 200

    tic = time.perf_counter()
    batch = pu.initialize_landmarks(songs, params, position, transition_matrix)
    toc = time.perf_counter()
    print('init landmarks', toc - tic)

    # calculus of the distance matrix (distance matrix for norm and vector, partition function )
    distances = pm.Distances(position, batch)

    # try 100, 200 iterations
    for i in range(params.n_iter):

        print('Iteration', i)

        if i % 10 == 0:
            tic = time.perf_counter()
            pu.update_landmarks(songs, batch, position, params)
            toc = time.perf_counter()
            print('Updated landmarks:', toc - tic, " seconds")

        # update the position of the songs
        tic = time.perf_counter()
        pu.update_songs(position_new, transition_matrix, position, params, distances)
        toc = time.perf_counter()

        print("Time to update songs:", toc - tic, " seconds")

        squared_error_new = (np.square(position_new - position)).mean(axis=None)
        # stop criteria
        if np.abs(squared_error_new - squared_error) < 1E-8 * squared_error:
            return position
        else:
            position_new, position = position, position_new
            squared_error = squared_error_new

        tic = time.perf_counter()
        distances.update(position, batch)
        toc = time.perf_counter()

        print("Time to update distances:", toc - tic, " seconds")

    return position_new
