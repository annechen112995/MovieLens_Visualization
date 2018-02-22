# Solution set for CS 155 Set 6, 2016/2017
# Authors: Fabian Boemer, Sid Murching, Suraj Nair

import numpy as np
from numpy import linalg as LA


def grad_U(Ui, Yij, Vj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    gradient = (reg * Ui) - (Vj * (Yij - np.dot(Ui, Vj)))
    ret = eta * gradient
    return ret


def grad_V(Vj, Yij, Ui, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point
    Yij, Ui (the ith row of U), reg (the regularization parameter lambda),
    and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    gradient = (reg * Vj) - ((Yij - np.dot(Ui, Vj)) * Ui)
    ret = eta * gradient
    return ret


def get_err(U, V, Y, reg, b, b_u, b_i, bias):
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
        if bias:
            pred = b + b_u[:,np.newaxis] + b_i[np.newaxis:,] + np.dot(U[i, :], V[j, :])
        else:
            pred = np.dot(U[i, :], V[j, :])
        dev = (Y_ij - pred)
        error += dev * dev

    normU = LA.norm(U, 'fro')
    normU_squared = normU * normU
    normV = LA.norm(V, 'fro')
    normV_squared = normV * normV

    error = ((reg / 2.) * (normU_squared + normV_squared)) + (
        (1 / 2.) * error)

    return error / n_rows


def train_model(M, N, K, eta, reg, Y, bias, eps=0.0001, max_epochs=300):
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
    print("K = %d" % K)

    n_rows = Y.shape[0]

    # Randomly initialize U and V
    U = np.random.rand(M, K) - 0.5
    V = np.random.rand(N, K) - 0.5
    print("U.shape: ", U.shape)
    print("V.shape: ", V.shape)

    # If adding bias
    ratings = Y[:, 2]
    if bias:
        b_u = np.zeros(M)
        b_i = np.zeros(N)
        b = np.mean(ratings[np.where(ratings != 0)])
    else:
        b = 0
        b_u = 0
        b_i = 0

    # Defines variable for keeping track with stopping condition
    prev_error = get_err(U, V, Y, reg, b, b_u, b_i, False)
    decrease = None

    for epoch in range(max_epochs):
        perm = np.random.permutation(n_rows)

        for row in perm:
            i = Y[row, 0] - 1
            j = Y[row, 1] - 1
            Y_ij = Y[row, 2]
            U[i, :] = U[i, :] - grad_U(U[i, :], Y_ij, V[j, :], reg, eta)
            V[j, :] = V[j, :] - grad_V(V[j, :], Y_ij, U[i, :], reg, eta)

            if bias:
                # Compute prediction and error
                prediction = b + b_u[i] + b_i[j] + U[i, :].dot(V[j, :].T)
                e = (Y_ij - prediction)

                # Update biases
                b_u[i] += eta * (e - reg * b_u[i])
                b_i[j] += eta * (e - reg * b_i[j])

        new_error = get_err(U, V, Y, reg, b, b_u, b_i, False)
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

    return (U, V, prev_error)
