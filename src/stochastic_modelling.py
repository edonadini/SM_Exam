import numpy as np
import random as rnd
import math as mt
from collections import namedtuple
import os
import pandas as pd


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
        # for j in range(len(dataset[row]) - 1):
        #    a = dataset[row][j];
        #    b = dataset[row][j + 1];
        #    count_matrix[a][b] = count_matrix[a][b] + 1
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
        while len(chunk[s]) / songs < params.r:
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
            # mi sa che devo considerare non Z ma delta perché voglio la distanza tra le canzoni
            # Z indica la partition function
            closest_idx = np.argmin(np.array([initial_distance[s][j] for j in landmarks]))
            closest_landmark = landmarks[closest_idx]

            if closest_landmark not in chunk[s]:
                chunk[s].append(closest_landmark)

    return chunk


def update_song_entry_vector(songs, transition_matrix, position_old, params, dist):
    position_new = np.empty_like(position_old)
    for s in range(songs):
        dev_term = np.array([transition_matrix[s, b] *
                             loss_derivative_on_entry(s, b, s, dist) for b in range(songs)])

        position_new[s] = position_old[s] + (params.tau / params.num_transition) * \
                          (sum(dev_term) - derivative_of_regularization_term_on_entry(position_old, s, params))

    return position_new


def log_like(test_set, probability_matrix):
    count = 0
    for i in range(len(test_set)):
        for predecessor, successor in zip(test_set[i][:-1], test_set[i][1:]):
            count = count + mt.log(probability_matrix[predecessor, successor])

    return count


def playlist_generator(num_song, current_song, song_hash, prob_matrix):
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
        playlist.append(np.random.choice(np.array(song_hash.iloc[0:7][0]), replace=True, p=prob_matrix[current_song]))
        prob = prob * prob_matrix[current_song, playlist[-1]]
        current_song = playlist[-1]

    # Printing the path of the Markov chain
    # Printing the number of steps
    # print(len(X)-1)
    titles = [song_hash.iloc[i][1] for i in playlist]
    print("Song in the playlist: ", titles)
    print("End state after ", num_song, " days: ", song_hash.iloc[current_song][1])
    print("Probability of the possible sequence of states: ", str(prob))

AlgParams = namedtuple('AlgParams', 'lam nu tau num_transition n_landmarks r dimension n_iter')


class Distances:

    def __init__(self, x, chunk):
        self.Z, self.D, self.diff = Distances.initialize_aux(x, chunk)

    def update(self, x, chunk):
        self.Z, self.D, self.diff = Distances.initialize_aux(x, chunk)

    @classmethod
    def initialize_aux(cls, x, chunk):
        d = Distances.delta(x)
        z = Distances.zeta(d, x, chunk)
        diff = Distances.difference_matrix(x)
        return z, d, diff

    @staticmethod
    def delta(x):
        dim = len(x)
        distance_mat = [[np.linalg.norm(x[i] - x[j]) for j in range(dim)] for i in range(dim)]
        return np.array(distance_mat).reshape((dim, dim))

    @staticmethod
    def zeta(d, x, chunk):
        z_vec = []
        for a in range(len(x)):
            sum_terms = np.array([d[a, landmark] for landmark in chunk[a]])
            z_vec.append(np.sum(np.exp(-(sum_terms ** 2))))
        return np.array(z_vec)

    @staticmethod
    def difference_matrix(x):
        dim = len(x)
        d = len(x[0])

        dif_mat = np.array([(x[i] - x[j]) for i in range(dim) for j in range(dim)])
        return dif_mat.reshape((dim, dim, d))


def loss_derivative_on_entry(a, b, p, dist):
    if a != p:
        return 0
    s_term = np.array([mt.exp(-dist.D[a, j] ** 2) * dist.diff[a, j, :] for j in range(len(dist.Z))])
    return 2 * (-dist.diff[a, b, :] + np.sum(s_term) / dist.Z[a])


def derivative_of_regularization_term_on_entry(x, p, params):
    return 2 * params.lam * x[p]


def single_point_algorithm(song_hash, transition_matrix, params):
    songs = len(song_hash)
    # assignment a random position at each song in the target space
    position = np.random.rand(songs, params.dimension)
    position_new = np.empty_like(position)

    # error for stopping criteria mean square error
    squared_error = 200

    # landmark initialization
    batch = initialize_landmarks(songs, params, position, transition_matrix)

    # calculus of the distance matrix (distance matrix for norm and vector, partition function )
    distances = Distances(position, batch)

    # try 100, 200 iterations
    for i in range(params.n_iter):
        print(i)
        # empirically update landmarks every 10 iterations
        # A iteration means a full pass on the training dataset.
        # fix the landmarks after 100 iteration to ensure convergence
        if i % 10 == 0 and i < 100:
            update_landmarks(songs, batch, distances, params)
        # update the position of the song in the space
        position_new = update_song_entry_vector(songs, transition_matrix, position, params, distances)
        squared_error_new = (np.square(position_new - position)).mean(axis=None)
        # stop criteria
        if abs(squared_error_new - squared_error) < 0.01 * squared_error:
            return position
        else:
            position_new, position = position, position_new
            squared_error = squared_error_new

    return position_new


root_dir = r'C:\Users\eleon\PycharmProjects\SM_Exam\data\yes_small'
# root_dir = './data/yes_small'


with open(os.path.join(root_dir, 'test.txt')) as f:
    # with open(root_dir + "\\test.txt", "r") as f:

    test_data = f.readlines()

test_dataset = data_to_list(test_data[2:])


with open(os.path.join(root_dir, "train.txt"), "r") as f:
    train_data = f.readlines()

train_dataset = data_to_list(train_data[2:])

song_hash = pd.read_csv(os.path.join(root_dir, "song_hash.txt"), sep="\t", header=None)


# setting parameters
dimension = 2  # rnd.choice([2, 5,10,25,50,100])
# regularization parameter set by cross validation
lam = 1#rnd.choice([0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 500, 1000])
# regularization parameter for dual point model
nu = rnd.choice([0, lam])
# threshold for landmark dimension
r = 0.2
# number of landmark
n_landmarks = 50
# num iteration 100 o 200
n_iter = 100  # rnd.choice([100, 200])
# tau predefined learning rate
tau = 0.5


# toy_dataset=[[0,6,3,2,1,4,2],[2,0,6,5,6,3,0],[0,5,2,1,6],[5,6,2],[6,5,2,3,5,6,2]]

songs = len(song_hash)
transition_matrix = transition_count(songs, train_dataset)
position = np.random.rand(songs, dimension)
num_transition = np.sum(transition_matrix)
params = AlgParams(lam, nu, tau, num_transition, n_landmarks, r, dimension, n_iter)

X = single_point_algorithm(song_hash, transition_matrix, params)


dummy_landmarks = [[i for i in range(songs)] for j in range(songs)]
d2 = Distances.delta(X)
prob_matrix = np.exp(-d2) / Distances.zeta(d2, X, dummy_landmarks)
cum_sum = np.sum(prob_matrix, axis=1)
prob_matrix = prob_matrix / cum_sum[:, np.newaxis]

# toy_test=[[0,4,5,3,6],[4,3,2,6],[0,2,1,0,5]]
# evaluate test performance using the average log-likelihood as metric
# it is defined as log(Pr(d_test))/n_test)
# n_test number of transition in the test set
# songs = 7 #len(song_hash)
n_test = np.sum(transition_count(songs, test_dataset))
print(n_test)

evaluation = log_like(test_dataset, prob_matrix) / n_test
print(evaluation)
# result = sum( sum(dimension.D[s, i.index()] ) for i in range(len(test_dataset)))


# generate a playlist
playlist_generator(4, 5, song_hash, prob_matrix)
