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

    # landmark initialization
    batch = pu.initialize_landmarks(songs, params, position, transition_matrix)

    # calculus of the distance matrix (distance matrix for norm and vector, partition function )
    distances = pm.Distances(position, batch)

    # try 100, 200 iterations
    for i in range(params.n_iter):
        print(i)
        # empirically update landmarks every 10 iterations
        # A iteration means a full pass on the training dataset.
        # fix the landmarks after 100 iteration to ensure convergence
        if i % 10 == 0 and i < 100:
            pu.update_landmarks(songs, batch, distances, params)

        # print('after update landmarks')
        # update the position of the song in the space
        position_new = pu.update_song_entry_vector(transition_matrix, position, params, distances)
        # print('after update vector')
        squared_error_new = (np.square(position_new - position)).mean(axis=None)
        # stop criteria
        if np.abs(squared_error_new - squared_error) < 1E-5 * squared_error:
            return position
        else:
            position_new, position = position, position_new
            squared_error = squared_error_new

        distances.update(position, batch)

    return position_new
