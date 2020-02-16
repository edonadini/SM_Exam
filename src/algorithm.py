import numpy as np
import pgmath as pm
import pgutils as pu


def single_point_algorithm(song_hash, dimension, train_dataset, n_iter, lam, nu, tau, n_landmarks, r):
    songs = 7#len(song_hash)
    # assignment a random position at each song in the target space
    position = np.random.rand(songs, dimension)
    position_new = np.empty_like(position)

    # error for stopping criteria mean square error
    squared_error = 200

    # number of transitions in the training set
    transition_matrix = pu.transition_count(songs, train_dataset)
    # total number of transition
    num_transition = np.sum(transition_matrix)
    # setting parameters
    params = pm.AlgParams(lam, nu, tau, num_transition, n_landmarks, r)

    # landmark initialization
    batch = pu.initialize_landmarks(songs, transition_matrix, params, position)

    # calculus of the distance matrix (distance matrix for norm and vector, partition function )
    space_position = pm.Distances(position, batch)

    # try 100, 200 iterations
    for i in range(n_iter):
        # empirically update landmarks every 10 iterations
        # A iteration means a full pass on the training dataset.
        # fix the landmarks after 100 iteration to ensure convergence
        if i % 10 == 0 and i < 100:
            pu.update_landmarks(songs, batch, space_position, params)
        # update the position of the song in the space
        position_new = pu.update_song_entry_vector(songs, transition_matrix, position, position_new, params,
                                                   space_position)
        squared_error_new = (np.square(position_new - position)).mean(axis=None)
        # stop criteria
        if abs(squared_error_new - squared_error) < 0.01 * squared_error:
            return position
        else:
            position_new, position = position, position_new
            squared_error = squared_error_new

    return position_new
