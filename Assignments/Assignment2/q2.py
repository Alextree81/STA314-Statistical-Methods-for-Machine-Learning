"""
STA314, 2021 Fall, University of Toronto
"""

import numpy as np

def robust_regression_grad(X, t, w, delta):
    """
    Compute the gradient of the average Huber (NOT squared error) loss for robust regression.

    Parameters
    ----------
        X: numpy array
            N x (D+1) numpy array for the train inputs (with dummy variables)
        t: numpy array
            N x 1 numpy array for the train targets
        w: numpy array
            (D+1) x 1 numpy array for the weights
        delta: positive float
            parameter for huber loss.


    Returns
    -------
        dw: numpy array
            (D+1) x 1 numpy array, the gradient of the huber loss in w
    """
    # ====== YOUR CODE GOES HERE (delete `pass') ======

    N, _ = np.shape(X)
    a = np.dot(X,w) - t

    dH = np.where(a > delta, delta,  np.where(a > -delta, a, -delta))


    dw = 1/N * np.dot(X.T, dH)

    return dw

    # =================================================



def optimization(X, t, delta, lr, num_iterations=10000):
    """
    Compute (nearly) optimal weights for robust linear regression.

    Parameters
    ----------
        X: numpy array
            N x (D+1) numpy array for the train inputs (with dummy variables)
        t: numpy array
            N x 1 numpy array for the train targets
        delta: positive float
            parameter for huber loss.
        lr: positive float
            learning rate or step-size.

    Returns
    -------
        w: numpy array
            (D+1) x 1 numpy array, (nearly) optimal weights for robust linear regression
    """

    # some initialization for robust regression parameters
    w = np.zeros((X.shape[1], 1))
    for i in range(num_iterations):
        # ====== YOUR CODE GOES HERE (delete `pass') ======
        w = w - lr*robust_regression_grad(X,t,w, delta)
        # =================================================
    return w

def squared_error(y, t):
    """
    Compute the average squared error (NOT huber) loss:

    sum_{i=1}^N (y^i - t^i)^2 / N

    Parameters
    ----------
        y: numpy array
            N x 1 numpy array for the predictions
        t: numpy array
            N x 1 numpy array for the train targets

    Returns
    -------
        cost: float
            the average squared error loss
    """
    dif = y - t
    L = 0.5 * np.power(dif, 2)
    return L.mean()

def linear_regression_optimal_weights(X, t):
    """
    Compute the optimal weights for linear regression.

    Parameters
    ----------
        X: numpy array
            N x (D+1) numpy array for the train inputs (with dummy variables)
        t: numpy array
            N x 1 numpy array for the train targets

    Returns
    -------
        w: numpy array
            (D+1) x 1 numpy array, optimal weights for linear regression
    """
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, t))
    return w


def main():
    # load data
    X = np.load("hw2_X.npy")
    t = np.load("hw2_t.npy")
    t = np.expand_dims(t, 1)
    (N, D) = X.shape

    # this code adds the "dummy variable"
    ones_vector = np.ones((N, 1))
    X = np.concatenate((ones_vector, X), axis=1)

    # train, validation, test split using numpy indexing
    X_train, X_val, X_test = X[30:], X[15:30], X[:15]
    t_train, t_val, t_test = t[30:], t[15:30], t[:15]

    lr = 0.01  # learning rate

    # these are the deltas we will try for robust regression
    deltas = [0.1, 0.5, 1, 5, 10]

    # Report the validation squared error loss using standard linear regression
    w_linreg = linear_regression_optimal_weights(X_train, t_train)
    predictions = np.dot(X_val, w_linreg)
    val_loss = squared_error(predictions, t_val)
    print(f"linear regression validation loss: {val_loss}")

    for i, delta in enumerate(deltas):

        # Optimize the parameters based on the train data for robust regression.
        w = optimization(X_train, t_train, delta, lr)

        # Report the validation and training squared error loss for this delta.
        val_pred = np.dot(X_val, w)
        val_loss = squared_error(t_val, val_pred)
        train_pred = np.dot(X_train, w)
        train_loss = squared_error(t_train, train_pred)
        print(f"delta: {delta}, valid. squared error loss: {val_loss}, train squared error loss: {train_loss}")


if __name__ == "__main__":
    main()













