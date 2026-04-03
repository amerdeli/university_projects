import numpy as np


def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)
    b = theta[0]
    w = theta[1]
    squared_error = (y - (b+w*x))**2
    return np.mean(squared_error)


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the 1D case of linear regression from the assigment sheet (see also slides from practicals)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    w = np.sum((x-x_mean)*(y-y_mean))/np.sum((x-x_mean)**2)
    b = y_mean - w*x_mean
    return np.array([b, w])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    # TODO: Implement Pearson correlation coefficient, as shown in Equation 3 (Task 1.1.1).
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    pearson_r = np.sum((x-x_mean)*(y-y_mean))/(np.sqrt(np.sum((x-x_mean)**2))*np.sqrt(np.sum((y-y_mean)**2)))
    #pearson_r = np.corrcoef(x,y)[0,1]
    return pearson_r


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)
    if data.ndim == 1:
        N = data.shape[0]
        D = 1
        data = data.reshape(-1,1)
    else:
        N = data.shape[0]
        D = data.shape[1]
    dummy_var = np.ones((N,1))
    design_matrix = np.concatenate((dummy_var, data), axis=1)
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)
    y_pred = X @ theta
    mse = np.mean((y_pred - y)**2)
    return float(mse)


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the solution to multivariate linear regression. 
    # Note: Use the pinv function for the Moore-Penrose pseudoinverse!
    theta = pinv(X) @ y
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)
    N = x.shape[0]
    polynomial_design_matrix = np.ones((N, K+1))

    for k in range(1, K+1):
        polynomial_design_matrix[:, k] = x ** k
        
    return polynomial_design_matrix