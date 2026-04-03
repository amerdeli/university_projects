import numpy as np


def create_design_matrix_dataset_1(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 1.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 1
    x1 = X_data[:,0].reshape(-1,1)
    x2 = X_data[:,1].reshape(-1,1)

    additional_features = np.concatenate((x1**2, x2**2, x1*x2),axis=1)
    X = np.concatenate((X_data, additional_features), axis=1)
    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_2(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 2.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 2
    x1 = X_data[:,0].reshape(-1,1)
    x2 = X_data[:,1].reshape(-1,1)

    additional_features = np.concatenate((x1**2, x2**2, x1*x2),axis=1)
    X = np.concatenate((X_data, additional_features), axis=1)
    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def create_design_matrix_dataset_3(X_data: np.ndarray) -> np.ndarray:
    """
    Create the design matrix X for dataset 3.
    :param X_data: 2D numpy array with the data points
    :return: Design matrix X
    """
    # TODO: Create the design matrix X for dataset 3
    x1 = X_data[:,0].reshape(-1,1)
    x2 = X_data[:,1].reshape(-1,1)

    additional_features = np.concatenate((x1**2, x1*x2, x2**2, x1**3, x1**2*x2, x2*x2**2, x2**3,
                                          x1**4, x1**3*x2, x1**2*x2**2, x1*x2**3, x2**4),axis=1)
    X = np.concatenate((X_data, additional_features), axis=1)
    assert X.shape[0] == X_data.shape[0], """The number of rows in the design matrix X should be the same as
                                             the number of data points."""
    assert X.shape[1] >= 2, "The design matrix X should have at least two columns (the original features)."
    return X


def logistic_regression_params_sklearn():
    """
    :return: Return a dictionary with the parameters to be used in the LogisticRegression model from sklearn.
    Read the docs at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """
    # TODO: Try different `penalty` parameters for the LogisticRegression model
    return {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter':100}