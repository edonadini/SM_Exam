# coding=utf-8

import numpy as np
import pgmath as pm
import pgutils as pu


def single_point_algorithm(songs, transition_matrix, params):
    # assignment a random position at each song in the target space
    position = np.random.rand(songs, params.dimension)
    position_new = np.empty_like(position)

    # error for stopping criteria mean square error
    squared_error = 200

    # calculus of the distance matrix (distance matrix for norm and vector, partition function )
    distances = pm.Distances(position)

    # try 100, 200 iterations
    for i in range(params.n_iter):
        print(i)

        # update the position of the songs
        position_new = pu.update_songs(transition_matrix, position, params, distances)

        squared_error_new = (np.square(position_new - position)).mean(axis=None)
        # stop criteria
        if np.abs(squared_error_new - squared_error) < 1E-5 * squared_error:
            return position
        else:
            position_new, position = position, position_new
            squared_error = squared_error_new

        distances.update(position)

    return position_new
