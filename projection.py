from scipy.linalg import svd
import numpy as np


def two_dir_projection(V):
    A, S, Bt = svd(V.T)

    # Best 2-directional projection of movies V
    proj = A[:, 0:2]
    return proj


def projection(U, V):
    proj = two_dir_projection(V)

    U_proj = np.dot(proj.T, U.T)
    V_proj = np.dot(proj.T, V.T)

    # Rescale so that each direction has variance of 1
    V_proj[0, :] /= np.std(V_proj[0, :])
    V_proj[1, :] /= np.std(V_proj[1, :])
    U_proj[0, :] /= np.std(U_proj[0, :])
    U_proj[1, :] /= np.std(U_proj[1, :])

    return U_proj, V_proj
