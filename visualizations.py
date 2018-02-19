import numpy as np
import csv

def loadRatings(fileName):
    '''
    Load data from the data.txt file

    Input format: user_id\tmovie_id\trating

    user_id = int
    movie_id = int
    rating = int
    '''
    ratings = []
    f = open(fileName, "r")

    for line in f:
        ratings.append(line.split())

    return np.asarray(ratings, dtype = int)

def loadMovies(fileName):
    '''
    Load data from the movies.txt file

    Input format: movie_id\tmovie_title\tUnknown\tAction\tAdventure\tAnimation
    \tChildrens\tComedy\tCrime\tDocumentary\tDrama\tFantasy\tFilm-Noir\tHorror
    \tMusical\tMystery\tRomance\tSci-Fi\tThriller\tWar\tWestern

    movie_id = int
    movie_title = string
    movie_genres = bool
    '''
    movies = []
    with open(fileName,'r') as f:
        reader=csv.reader(f,delimiter='\t')
        for movieData in reader:
            movies.append(movieData)
    return np.array(movies)

def main():
    movie_ratings = loadRatings("data.txt")
    movies = loadMovies("movies.txt")
