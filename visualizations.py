import csv
import numpy as np
import matrixFactorization as mf
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from basicVisualization import basicVisualization
from projection import projection

BORDER = "==================================================================="


def loadRatings(fileName):
    '''
    Load data from the data.txt file

    Input format: user_id\tmovie_id\trating

    user_id = int
    movie_id = int
    rating = int
    '''
    ratings = []
    f = open(fileName, 'r')

    for line in f:
        ratings.append(line.split())

    return np.asarray(ratings, dtype=int)


def loadMovies(fileName):
    '''
    Load data from the movies.txt file

    Input format: movie_id\tmovie_title\tUnknown\tAction\tAdventure\tAnimation
    \tChildrens\tComedy\tCrime\tDocumentary\tDrama\tFantasy\tFilm-Noir\tHorror
    \tMusical\tMystery\tRomance\tSci-Fi\tThriller\tWar\tWestern

    movie_id = int
    movie_title = string
    {Unknown, Action,..., Western} = bool (0 or 1)

    returns three dictionaries
    movie_ID = {movie_id:movie_title}
    movie_categoty = {movie_id:categories}
        where categories is a numpy array of 0 or 1 representing if the
        movie falls under the category
    movie_genres = {index of genre: genre name}
    '''
    movie_ID = {}
    movie_category = {}

    movie_genres = {
        0: 'Unknown', 1: 'Action', 2: 'Adventure', 3: 'Animation',
        4: 'Childrens', 5: 'Comedy', 6: 'Crime', 7: 'Documentary',
        8: 'Drama', 9: 'Fantasy', 10: 'Film-Noir', 11: 'Horror',
        12: 'Musical', 13: 'Mystery', 14: 'Romance', 15: 'Sci-Fi',
        16: 'Thriller', 17: 'War', 18: 'Western'}

    with open(fileName, encoding='ISO-8859-1') as f:
        reader = csv.reader(f, delimiter='\t')
        for movieData in reader:
            # print(movieData)
            movie_ID[int(movieData[0])] = movieData[1]
            categories = [int(x) for x in movieData[2:]]
            movie_category[int(movieData[0])] = np.asarray(categories)

    return movie_ID, movie_category, movie_genres


def method1_reg(train, test):
    print(BORDER)
    print("Method 1 - reg")
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)  # users
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int)  # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20

    regs = [10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2]
    eta = 0.03  # learning rate
    E_ins = []
    E_outs = []

    for reg in regs:
        print("Training model with reg = %s" % (reg))
        U, V, _ = mf.train_model(M, N, K, eta, reg, train)
        E_ins.append(mf.get_err(U, V, train))
        E_outs.append(mf.get_err(U, V, test))

    plt.plot(regs, E_ins, label='$E_{in}$')
    plt.plot(regs, E_outs, label='$E_{out}$')
    plt.title('Error vs. reg')
    plt.xlabel('reg')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.legend()
    plt.savefig('method1_reg.png')


def method1(train, test):
    print(BORDER)
    print("Method 1")
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)  # users
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int)  # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20

    reg = 10**-1
    eta = 0.03  # learning rate
    U, V, _ = mf.train_model(M, N, K, eta, reg, train)
    E_in = mf.get_err(U, V, train)
    E_out = mf.get_err(U, V, test)
    print("E_in = ", E_in)
    print("E_out = ", E_out)

    return U, V


def get_err_from_pred(pred, ratings):
    n_rows = ratings.shape[0]
    error = 0

    for row in range(n_rows):
        user_ind = ratings[row, 0] - 1
        movie_ind = ratings[row, 1] - 1
        rating = ratings[row, 2]
        dev = rating - pred[user_ind, movie_ind]
        error += dev * dev
    return ((1 / 2.) * error) / n_rows


def method3(train, test):
    print(BORDER)
    print("Method 3")
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)  # users
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int)  # movies

    train_matrix = np.zeros((M, N))
    n_rows = train.shape[0]
    for row in range(n_rows):
        user_ind = train[row, 0] - 1
        movie_ind = train[row, 1] - 1
        train_matrix[user_ind, movie_ind] = train[row, 2]
    U, s, Vt = svds(train_matrix, k=20)
    s_diag_matrix = np.diag(s)
    pred = np.dot(np.dot(U, s_diag_matrix), Vt)
    E_in = get_err_from_pred(pred, train)
    E_out = get_err_from_pred(pred, test)
    print("E_in = ", E_in)
    print("E_out = ", E_out)

    return U, Vt.transpose()


if __name__ == '__main__':
    directory = 'visualizations/'
    dataFile = "data/data.txt"
    moviesFile = "data/movies.txt"
    trainingFile = "data/train.txt"
    testFile = "data/test.txt"

    # all movie ratings as movie_ratings = [[user ID, movie ID, rating]]
    movie_ratings = loadRatings(dataFile)
    # movie info as three dictionaries:
    # movie_ID = {movie_id:movie_title}
    # movie_category = {movie_id:categories}
    # movie_genres = {index of genre: genre name}
    movie_ID, movie_category, movie_genres = loadMovies(moviesFile)

    basicVisualization(movie_ratings, movie_ID, movie_category, movie_genres,
                       directory)

    train = loadRatings(trainingFile)
    test = loadRatings(testFile)

    U, V = method1(train, test)
    # method3(train, test)

    projection(V)
