from scipy.linalg import svd
import numpy as np


def two_dir_projection(V):
    A, S, Bt = svd(V.T)

    # Best 2-directional projection of movies V
    proj = A[:, 0:2]
    return proj


def projection(U, V):
    proj = two_dir_projection(V)

    movie_proj = np.dot(proj.T, V.T)
    user_proj = np.dot(proj.T, U.T)

    # Rescale so that each direction has variance of 1
    movie_proj[0, :] /= np.std(movie_proj[0, :])
    movie_proj[1, :] /= np.std(movie_proj[1, :])
    user_proj[0, :] /= np.std(user_proj[0, :])
    user_proj[1, :] /= np.std(user_proj[1, :])

    return movie_proj, user_proj
