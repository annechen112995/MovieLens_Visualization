# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
from numpy import linalg as LA


def grad_U(Ui, Yij, Vj, ai, bj, mu, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    t1 = Yij - mu - np.dot(Ui, Vj) - ai - bj
    gradient = (reg * Ui) - (2 * (Vj * (t1)))
    ret = eta * gradient
    return ret


def grad_V(Vj, Yij, Ui, ai, bj, mu, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point
    Yij, Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    t1 = Yij - mu - np.dot(Ui, Vj) - ai - bj
    gradient = (reg * Vj) - (2 * (Ui * (t1)))
    ret = eta * gradient
    return ret


def grad_A(Ui, Yij, Vj, ai, bj, mu, reg, eta):
    t1 = Yij - mu - np.dot(Ui, Vj) - ai - bj
    gradient = (reg * ai) - 2*(t1)
    ret = eta * gradient
    return ret


def grad_B(Ui, Yij, Vj, ai, bj, mu, reg, eta):
    t1 = Yij - mu - np.dot(Ui, Vj) - ai - bj
    gradient = (reg * bj) - 2*(t1)
    ret = eta * gradient
    return ret


def get_err(U, V, Y, a, b, mu, reg):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of
    a user, j is the index of a movie, and Y_ij is user i's rating of movie j
    and user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth
    column of V^T.
    """
    n_rows = Y.shape[0]
    error = 0

    for row in range(n_rows):
        i = Y[row, 0] - 1
        j = Y[row, 1] - 1
        Y_ij = Y[row, 2]
        pred = np.dot(U[i, :], V[j, :])
        dev = (Y_ij - pred - mu - a[i] - b[j])
        error += dev * dev

    normU = LA.norm(U, 'fro')
    normU_squared = normU * normU
    normV = LA.norm(V, 'fro')
    normV_squared = normV * normV
    normA = LA.norm(a)
    normA_squared = normA * normA
    normB = LA.norm(b)
    normB_squared = normB * normB

    error = ((reg / 2.) * (normU_squared + normV_squared + normA_squared +
        normB_squared)) + error

    return error / n_rows


def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    # print("K = %d" % K)

    n_rows = Y.shape[0]

    # Randomly initialize U and V
    U = np.random.rand(M, K) - 0.5
    V = np.random.rand(N, K) - 0.5
    # print("U.shape: ", U.shape)
    # print("V.shape: ", V.shape)

    # Generate bias matrix
    ratings = Y[:, 2]
    a = np.zeros(M)
    b = np.zeros(N)
    mu = np.mean(ratings[np.where(ratings != 0)])

    # Defines variable for keeping track with stopping condition
    prev_error = get_err(U, V, Y, a, b, mu, reg)
    decrease = None

    for epoch in range(max_epochs):
        perm = np.random.permutation(n_rows)

        for row in perm:
            i = Y[row, 0] - 1
            j = Y[row, 1] - 1
            Y_ij = Y[row, 2]

            # Update biases
            a[i] = a[i] - grad_A(U[i, :], Y_ij, V[j, :], a[i], b[j], mu, reg, eta)
            b[j] = b[j] - grad_B(U[i, :], Y_ij, V[j, :], a[i], b[j], mu, reg, eta)

            U[i, :] = U[i, :] - grad_U(U[i, :], Y_ij, V[j, :], a[i], b[j], mu, reg, eta)
            V[j, :] = V[j, :] - grad_V(V[j, :], Y_ij, U[i, :], a[i], b[j], mu, reg, eta)

        new_error = get_err(U, V, Y, a, b, mu, reg)
        print("epoch: %d, error: %f" % (epoch, new_error))

        # Epoch 1
        if decrease is None:
            decrease = np.fabs(prev_error - new_error)
            prev_error = new_error
            continue

        # Remaining epochs
        if np.fabs(new_error - prev_error) <= eps * decrease:
            prev_error = new_error
            break
        else:
            prev_error = new_error

    return (U, V, prev_error, a, b, mu)
