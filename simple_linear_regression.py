import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generating_data(n, m=2.25, b=6.0, stddev=1.5):
    x = np.linspace(-2.0, 2.0, n)
    y = x * m + b + np.random.normal(loc=0, scale=stddev, size=n)
    return x, y

def compute_cost(X, Y, m, c):
    """Computes the cost of the function for the given parameters. 
    
    Input:
    X, Y - refers to the input and the output array
    m, c - refers to the slope and intercept respectively

    Output:
    Returns Mean Square Error.
    """
    
    Yhat = m * X + c
    diff = Yhat - Y
    mean_square_error = np.dot(diff.T, diff) / X.shape[0]
    return mean_square_error.flat[0]


def gradient_descent(X, Y, nsteps, step_size=0.0001):
    """Returns the minimal value of the parameters m, c after running for nsteps. 
    
    Input:
    X, Y - refers to the input and the output array.
    nsteps - refers to how many times the optimization should run for
    step_size - refers to the learning rate of the algorithm
    """

    m, c = 0, 0
    yield m, c, compute_cost(X, Y, m, c)
    # print(m, c, compute_cost(X, Y, m, c))
    for i in range(nsteps):
        Yhat = m * X + c
        diff = Yhat - Y
        dm = step_size * (diff * X).sum() * 2 / X.shape[0]
        dc = step_size *  diff.sum() * 2 / X.shape[0]
        m -= dm
        c -= dc
        # print(m, c, compute_cost(X, Y, m, c))
        yield m, c, compute_cost(X, Y, m, c)

X, Y = pd.Series([95, 85, 80, 70, 60]), pd.Series([85, 95, 70, 65, 70])
for i in iter(gradient_descent(X, Y, 1000)):
    print(i)