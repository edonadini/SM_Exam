import utilitymain as sm
import sys

#directory = r'C:\Users\eleon\PycharmProjects\SM_Exam\data\yes_small'
dimensions = [2, 5, 10, 25, 50, 100]
lams = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 500, 1000]
rs = [0.1, 0.2, 0.3, 0.4, 0.5]
n_iters = [100, 200]
n_song = 4
start_song = 5


dimension = int(sys.argv[1])
lam = float(sys.argv[2])
r = float(sys.argv[3])
n_iter = int(sys.argv[4])
n_song = int(sys.argv[5])
start_song = int(sys.argv[6])
directory = sys.argv[7]

sm.test(directory, dimension, lam, r, n_iter, n_song, start_song)