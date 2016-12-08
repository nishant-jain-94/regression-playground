import numpy as np

def generating_data(n, m=2.25, b=6.0, stddev=1.5):
    x = np.linspace(-2.0, 2.0, n)
    y = x * m + b
    return x, y

def compute_cost(X, Y, m, c):
    """Computes the cost of the function for the given parameters. 
    
    Input:
    X, Y - refers to the input and the output array
    m, c - refers to the slope and intercept respectively

    Output:
    Returns Residual Sum of Squares.
    """
    
    Yhat = m * X + c
    diff = Yhat - Y
    residual_sum_of_squares = np.dot(diff.T, diff).sum()
    return residual_sum_of_squares


def gradient_descent(X, Y, nsteps, step_size=0.1):
    """Returns the minimal value of the parameters m, c after running for nsteps. 
    
    Input:
    X, Y - refers to the input and the output array.
    nsteps - refers to how many times the optimization should run for
    step_size - refers to the learning rate of the algorithm
    """

    m, c = 0, 0
    yield m, b, compute_cost(X, Y, m, c)
    for i in range(nsteps):
        Yhat = m * X + c
        diff = Yhat - Y
        dm = step_size * (2 * diff) * X
        dc = step_size * (2 * diff)
        m -= dm
        c -= dc
        yield m, c, compute_cost(X, Y, m, c)


generating_data(50)