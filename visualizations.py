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

    return np.asarray(ratings, dtype=int)



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
    movies_new = []

    movie_genres = {2: "Unknown", 3: "Action", 4: "Adventure", 5: "Animation",
            6: "Childrens", 7: "Comedy", 8: "Crime", 9: "Documentary",
            10: "Drama", 11: "Fantasy", 12: "Film-Noir", 13: "Horror",
            14: "Musical", 15: "Mystery", 16: "Romance", 17: "Sci-Fi",
            18: "Thriller", 19: "War", 20:"Western"}

    with open(fileName, encoding='ISO-8859-1') as f:
        reader=csv.reader(f,delimiter='\t')
        for movieData in reader:
            movies.append(movieData)

    movies = np.asarray(movies)

    for movie in movies:
        index = -1
        for i, j in enumerate(movie):
            if j == '1':
                index = i
        movies_new.append([movie[0], movie[1], movie_genres[index]])

    return movies_new


if __name__ == "__main__":
    movie_ratings = loadRatings("data/data.txt")
    print("movie_ratings: ", movie_ratings)
    movies = loadMovies("data/movies.txt")
    print("movies: ", movies)
