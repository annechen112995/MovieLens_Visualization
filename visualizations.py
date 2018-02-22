import csv
import numpy as np
import matrixFactorization as mf
import biasMatrixFactorization as bmf
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
        0:  'Unknown',   1:  'Action',  2:  'Adventure', 3: 'Animation',
        4:  'Childrens', 5:  'Comedy',  6:  'Crime',     7: 'Documentary',
        8:  'Drama',     9:  'Fantasy', 10: 'Film-Noir', 11: 'Horror',
        12: 'Musical',   13: 'Mystery', 14: 'Romance',   15: 'Sci-Fi',
        16: 'Thriller',  17: 'War',     18: 'Western'}

    with open(fileName, encoding='ISO-8859-1') as f:
        reader = csv.reader(f, delimiter='\t')
        for movieData in reader:
            # print(movieData)
            movie_ID[int(movieData[0])] = movieData[1]
            categories = [int(x) for x in movieData[2:]]
            movie_category[int(movieData[0])] = np.asarray(categories)

    return movie_ID, movie_category, movie_genres


def Homework_5_SVD_With_Regularization(train, test):
    print(BORDER)
    print("Homework_5_SVD_With_Regularization - reg")
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


def Homework_5_SVD(train, test):
    print(BORDER)
    print("Homework_5_SVD")
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)  # users
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int)  # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20

    reg = 10**-1
    eta = 0.03  # learning rate
    U, V, _ = mf.train_model(M, N, K, eta, reg, train)
    E_in = mf.get_err(U, V, train, 0.0)
    E_out = mf.get_err(U, V, test, 0.0)
    print("E_in = ", E_in)
    print("E_out = ", E_out)

    return U, V


def SVD_With_Bias(train, test):
    print(BORDER)
    print("SVD_With_Bias")
    M = max(max(train[:, 0]), max(test[:, 0])).astype(int)  # users
    N = max(max(train[:, 1]), max(test[:, 1])).astype(int)  # movies
    print("Factorizing with ", M, " users, ", N, " movies.")
    K = 20

    reg = 10**-1
    eta = 0.03  # learning rate
    U, V, err, a, b, mu = bmf.train_model(M, N, K, eta, reg, train)
    E_in = bmf.get_err(U, V, train, a, b, mu, reg)
    E_out = bmf.get_err(U, V, test, a, b, mu, reg)
    print("E_in = ", E_in)
    print("E_out = ", E_out)

    return U, V


def Get_Err_From_Pred(pred, ratings):
    n_rows = ratings.shape[0]
    error = 0

    for row in range(n_rows):
        user_ind = ratings[row, 0] - 1
        movie_ind = ratings[row, 1] - 1
        rating = ratings[row, 2]
        dev = rating - pred[user_ind, movie_ind]
        error += dev * dev
    return ((1 / 2.) * error) / n_rows


def Off_The_Shelf_SVD(train, test):
    print(BORDER)
    print("Off_The_Shelf_SVD")
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
    E_in = Get_Err_From_Pred(pred, train)
    E_out = Get_Err_From_Pred(pred, test)
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
    test  = loadRatings(testFile)

    U, V = Homework_5_SVD(train, test)

    U_proj, V_proj = projection(U, V)

    #===============================================================================================
    # Visualize V for any ten movies of your choice from the MovieLens dataset.
    # Let's choose the first 10, because I'm lazy
    #===============================================================================================
    V0 = V_proj[0, 1:10]
    V1 = V_proj[1, 1:10]

    #===============================================================================================
    # Visualize V for the ten most popular movies (movies which have received the most ratings).
    #===============================================================================================
	
    #===============================================================================================
	# Visualize V for the ten best movies (movies with the highest average ratings).
    #===============================================================================================
	
    #===============================================================================================
	# Visualize V for ten movies from the Comedy genre you selected in Section 4, Basic Visualizations
    #===============================================================================================
	
    #===============================================================================================
	# Visualize V for ten movies from the Romance genre you selected in Section 4, Basic Visualizations
    #===============================================================================================

    #===============================================================================================
	# Visualize V for ten movies from the Horror genre you selected in Section 4, Basic Visualizations
    #===============================================================================================
