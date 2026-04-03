import numpy as np
from typing import Callable, Tuple


def gradient_descent(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                     df: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                     x0: np.ndarray, 
                     y0: np.ndarray, 
                     learning_rate: float, 
                     lr_decay: float, 
                     num_iters: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find a local minimum of the function f(x, y) using gradient descent:
    Until the number of iteration is reached, decrease the current x and y points by the
    respective partial derivative times the learning_rate.
    In each iteration, record the current function value in the list f_list 
    and the current x and y points in the lists x_list and y_list.
    The function should return the lists x_list, y_list, f_list.

    :param f: Function to minimize
    :param df: Gradient of f i.e, function that computes gradients
    :param x0: initial x0 point
    :param y0: initial y0 point
    :param learning_rate: Learning rate
    :param lr_decay: Learning rate decay
    :param num_iters: number of iterations
    :return: x_list, y_list, f_list (lists of x, y, and f values over iterations). 
             The first element of the lists represents the initial point (and the function value at the initial point).
             The last element of the lists represents the final point (and the function value at the final point).
    """
    f_list = np.zeros(num_iters+1)
    x_list = np.zeros(num_iters+1)
    y_list = np.zeros(num_iters+1)

    # TODO: Implement the gradient descent algorithm with a decaying learning rate

    # add initial point values to the lists
    x_list[0], y_list[0], f_list[0]= x0, y0, f(x0, y0)

    # set initial step size 
    eta = learning_rate

    for i in range(1,num_iters+1):

        # calculate gradient values at the current point
        df_dx, df_dy = df(x_list[i-1], y_list[i-1])

        # update x,y values
        x_list[i] = x_list[i-1] - eta*df_dx
        y_list[i] = y_list[i-1] - eta*df_dy

        # update function value
        f_list[i] = f(x_list[i], y_list[i])

        # update step size
        eta = lr_decay*eta

        #print(f"Old function value f = {f_list[i-1]}, new function value f = {f_list[i]}")
    return x_list, y_list, f_list


def rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements the Rastrigin function (as specified in the assignment sheet)
    :param x: x-coordinate
    :param y: y-coordinate
    :return: Rastrigin function value
    """
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


def gradient_rastrigin(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Implements partial derivatives of Rastrigin function w.r.t. x and y
    :param x: x-coordinate
    :param y: y-coordinate
    :return: Gradient of Rastrigin function
    """
    df_dx = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    df_dy = 2 * y + 20 * np.pi * np.sin(2 * np.pi * y)

    gradient = np.array([df_dx, df_dy])
    return gradient
