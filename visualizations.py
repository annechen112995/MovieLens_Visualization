import csv
import heapq
import itertools
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from collections import Counter


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


def getPopularMovies(movie_ratings, movies):
    '''
    Return the top ten most popular movies (most rated) and their ratings.
    '''
    popularMovies = list(movie_ratings[:, 1])
    count = Counter(popularMovies)
    topTen = count.most_common(10)

    return topTen


def getMovieRatings(movie_ratings):
    '''
    Returns a dictionary of {movie_id: np.array([ratings])}
    '''
    # dictionary of ratings
    ratings = {}
    for _, movie_id, rating in movie_ratings:
        if movie_id not in ratings:
            ratings[movie_id] = np.array([rating])
        else:
            ratings[movie_id] = np.append(ratings[movie_id], rating)
    return ratings


def getBestMovies(movie_ratings):
    '''
    Return the top ten most highly rated movies and their ratings.
    '''
    num_highest = 10

    # dictionary of ratings
    ratings = getMovieRatings(movie_ratings)

    # average ratings
    avg_ratings = {k: np.mean(ratings[k]) for k in ratings}
    sorted_avg_ratings = sorted(avg_ratings.items(), key=itemgetter(1),
                                reverse=True)

    highest_avg_movie = sorted_avg_ratings[:num_highest]
    highest_avg_movie_ratings = {k: ratings[k] for (k, _) in highest_avg_movie}

    return highest_avg_movie_ratings


def getThreeGenres(movie_category, genreList):
    '''
    Return all movies in three genres (comedy, horror, romance) and their
    ratings.
    '''
    # Index corresponding to genres Comedy, Horror, Romance
    genreMoviesList = []
    for i in range(len(genreList)):
        movieList = []
        for j in range(len(movie_category)):
            if (movie_category[j + 1][genreList[i]]) == 1:
                movieList.append(j + 1)
        genreMoviesList.append(movieList)

    return genreMoviesList


def allRatingsPlot(movie_ratings, directory, title):
    '''
    Plot all ratings in MovieLens dataset
    '''
    ratings = movie_ratings[:, 2]

    # Plot Histogram
    hist, _ = np.histogram(ratings, bins=[1, 2, 3, 4, 5, 6])
    plt.bar(np.arange(1, 6), hist, align='center')
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Num. Movies')
    plt.savefig(directory + title + '_Histogram' + '.png', bbox_inches='tight')
    plt.clf()


def popularRatingsPlot(movie_ratings, movies, directory, title):
    '''
    Plot all ratings of ten most popular movies
    '''
    popularMovies = getPopularMovies(movie_ratings, movies)
    popularMovieIds = [i[0] for i in popularMovies]
    topTenRatings = []

    for i in range(len(popularMovieIds)):
        ratingsList = []
        for j in range(len(movie_ratings[:, 1])):
            if movie_ratings[j, 1] == popularMovieIds[i]:
                ratingsList.append(movie_ratings[j, 2])
        topTenRatings.append(ratingsList)

    # Plot Histogram
    plt.style.use('seaborn-deep')
    plt.hist(
        [topTenRatings[0], topTenRatings[1], topTenRatings[2], topTenRatings[3],
        topTenRatings[4], topTenRatings[5], topTenRatings[6], topTenRatings[7],
        topTenRatings[8], topTenRatings[9]], bins=[1, 2, 3, 4, 5, 6],
        label = [popularMovieIds[0], popularMovieIds[1], popularMovieIds[2],
        popularMovieIds[3], popularMovieIds[4], popularMovieIds[5],
        popularMovieIds[6], popularMovieIds[7], popularMovieIds[8],
        popularMovieIds[9]])
    plt.title(title)
    plt.legend(loc='upper left')
    plt.xlabel('Rating')
    plt.ylabel('Num. Ratings')
    plt.savefig(directory + title + '_Histogram' + '.png', bbox_inches='tight')
    plt.clf()


def bestRatingsPlot(movie_ratings, directory, title):
    '''
    Plot all ratings of ten best movies (highest avg. ratings)
    '''

    # Get {movie_id: np.array([ratings])} of 10 top rated movies
    bestMovies = getBestMovies(movie_ratings)

    # Get the list of ratings of these movies
    ratingsOfBestMovies = []
    for ratings in bestMovies.values():
        ratingsOfBestMovies += list(ratings)

    # Plot Histogram
    hist, _ = np.histogram(ratingsOfBestMovies, bins=[1, 2, 3, 4, 5, 6])
    plt.bar(np.arange(1, 6), hist, align='center')
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Num. Movies')
    plt.savefig(directory + title + '_Histogram' + '.png', bbox_inches='tight')
    plt.clf()


def genreRatingsPlot(movie_ratings, movie_ID, movie_category, movie_genres,
    directory, title):
    '''
    Plot all ratings from three genres
    '''
    genreList = [5, 11, 14]
    genreMovies = getThreeGenres(movie_category, genreList)
    movieRatings = getMovieRatings(movie_ratings)
    totalRatingsList = []

    for i in range(len(genreMovies)):
        ratingsList = []
        for j in range(len(genreMovies[i])):
            index = int(genreMovies[i][j])
            ratingsList = ratingsList + list(movieRatings[index])
        totalRatingsList.append(ratingsList)

    for k in range(len(totalRatingsList)):
        # Plot Histogram
        hist, _ = np.histogram(totalRatingsList[k], bins=[1, 2, 3, 4, 5, 6])
        plt.bar(np.arange(1, 6), hist, align='center')
        plt.title(movie_genres[genreList[k]] + '_' + title)
        plt.xlabel('Rating')
        plt.ylabel('Num. Movies')
        plt.savefig(
            directory + movie_genres[genreList[k]] + '_' + title + '_Histogram'
            + '.png', bbox_inches='tight')
        plt.clf()


if __name__ == '__main__':
    # movie_ratings = [[user ID, movie ID, rating]]
    movie_ratings = loadRatings('data/data.txt')
    # movie_ID = {movie_id:movie_title}
    # movie_category = {movie_id:categories}
    # movie_genres = {index of genre: genre name}
    movie_ID, movie_category, movie_genres = loadMovies('data/movies.txt')

    directory = 'visualizations/'
    allRatingsTitle = 'All_Ratings'
    popularRatingsTitle = 'Top_Ten_Popular_Movie_Ratings'
    bestRatingsTitle = 'Top_Ten_Best_Movie_Ratings'
    genreRatingsTitle = 'Movie_Ratings'

    # Plotting all ratings in MovieLens dataset
    allRatingsPlot(movie_ratings, directory, allRatingsTitle)

    # Plotting all ratings of ten most popular movies
    popularRatingsPlot(movie_ratings, movie_ID, directory, popularRatingsTitle)

    # Plotting all ratings of ten best movies (highest avg. ratings)
    bestRatingsPlot(movie_ratings, directory, bestRatingsTitle)

    # Plotting all ratings from three genres
    genreRatingsPlot(
        movie_ratings, movie_ID, movie_category, movie_genres, directory,
        genreRatingsTitle)
