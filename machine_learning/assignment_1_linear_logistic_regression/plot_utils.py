from matplotlib import pyplot as plt
import numpy as np
from typing import Callable
from sklearn.inspection import DecisionBoundaryDisplay
from linear_regression import compute_polynomial_design_matrix

def pairplot(data: np.ndarray, feature_names: list[str]) -> None:
    """
    Create a pairplot of the given data.
    :param data: 2D array of shape (N, D) that represents the data matrix
    :param feature_names: List of feature names
    :return: None
    """
    num_features = data.shape[1]
    fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                axes[i, j].hist(data[:, i], bins=20, color='gray', alpha=0.7)
                axes[i, j].set_title(feature_names[i])
            elif i > j:
                axes[i, j].scatter(data[:, j], data[:, i], alpha=0.5, s=3)
            else:
                # Upper triangle: hide plots
                axes[i, j].set_visible(False)
            if i == num_features - 1:
                axes[i, j].set_xlabel(feature_names[j])
            if j == 0:
                axes[i, j].set_ylabel(feature_names[i])
    plt.tight_layout()
    plt.show()


def plot_scatterplot_and_polynomial(x: np.ndarray, 
                                    y: np.ndarray, 
                                    theta: np.ndarray, 
                                    xlabel: str = 'x', 
                                    ylabel: str = 'y', 
                                    title: str = 'Title', 
                                    figname: str = 'scatterplot_and_polynomial') -> None:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param title: Title of the plot
    :param figname: Filename to save the plot in the `plots` directory
    :return:
    """
    # Theta will be an array with two coefficients, representing slope and intercept.
    # In which format is it stored in the theta array? Take care of that when plotting the line.
    plt.scatter(x, y)
    xx = np.linspace(np.min(x), np.max(x), 100)
    X = compute_polynomial_design_matrix(xx, K=len(theta)-1)
    yy = X @ theta
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(xx, yy, color='orange')

    plt.savefig(f'plots/{figname}.pdf')
    plt.show()


def plot_scatterplot_and_line(x: np.ndarray, 
                              y: np.ndarray, 
                              theta: np.ndarray, 
                              xlabel: str = 'x', 
                              ylabel: str = 'y', 
                              title: str = 'Title', 
                              figname: str = 'scatterplot_and_line') -> None:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :param xlabel: Label for the x-axis
    :param ylabel: Label for the y-axis
    :param title: Title of the plot
    :param figname: Filename to save the plot in the `plots` directory
    :return:
    """
    # Theta will be an array with two coefficients, representing slope and intercept.
    # In which format is it stored in the theta array? Take care of that when plotting the line.
    plt.scatter(x, y)
    b, w = theta[0], theta[1]
    x_line = np.array([np.min(x), np.max(x)])
    y_line = w * x_line + b
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x_line, y_line, color='orange')

    plt.savefig(f'plots/{figname}.pdf')
    plt.show()


def plot_logistic_regression(logreg_model, create_design_matrix, X, 
                             title: str, figname: str) -> None:
    """
    Plot the decision boundary of a logistic regression model.
    :param logreg_model: The logistic regression model
    :param create_design_matrix: Function to create the design matrix
    :param X: Data matrix
    :param title: Title of the plot
    :param figname: Filename to save the plot in the `plots` directory
    :return:
    """
    y = logreg_model.predict(X)
    xx0, xx1 = np.meshgrid(
        np.linspace(np.min(X[:, 0]), np.max(X[:, 0])),
        np.linspace(np.min(X[:, 1]), np.max(X[:, 1]))
    )
    x_grid = np.vstack([xx0.reshape(-1), xx1.reshape(-1)]).T
    x_grid = create_design_matrix(x_grid)
    y_grid = logreg_model.predict(x_grid).reshape(xx0.shape)
    display = DecisionBoundaryDisplay(xx0=xx0, xx1=xx1, response=y_grid)

    display.plot()
    p = display.ax_.scatter(
        X[:, 0], X[:, 1], c=y, edgecolor="black"
    )

    display.ax_.set_title(title)
    display.ax_.collections[0].set_cmap('coolwarm')
    display.ax_.figure.set_size_inches(5, 5)
    display.ax_.set_xlabel('x1')
    display.ax_.set_ylabel('x2')
    display.ax_.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(1.02, 1.15))

    # make sure a "plots" directory exists!
    plt.savefig(f'plots/{figname}.pdf')
    plt.show()


def plot_datapoints(X: np.ndarray, y: np.ndarray, title: str) -> None:
    """
    Plot the data points in a scatter plot with color-coded classes.
    :param X: The data points
    :param y: The class labels
    :param title: Title of the plot
    :return:
    """
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    fig.suptitle(title, y=0.93)

    p = axs.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

    axs.set_xlabel('x1')
    axs.set_ylabel('x2')
    axs.legend(*p.legend_elements(), loc='best', bbox_to_anchor=(0.96, 1.15))

    plt.show()


def plot_3d_surface(f: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
    """
    Plotting the 3D surface for a given cost function f.
    :param f: The function to optimize
    :return:
    """
    n = 500
    bounds = [-2, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = f(XX, YY)

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def plot_2d_contour(f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                    starting_point: np.ndarray = None, 
                    global_min: np.ndarray = None, 
                    x_list: np.ndarray = None, 
                    y_list: np.ndarray = None, 
                    figname: str = '2d_contour') -> None:
    """
    Plot the 2D contour of a given function f.
    :param f: The function to plot
    :param starting_point: A point that will be highlighted in the contour plot
    :param global_min: The global minimum of the function
    :param x_list: The list of x values
    :param y_list: The list of y values
    :param figname: Filename to save the plot in the `plots` directory
    :return:
    """
    n = 500
    if x_list is not None and y_list is not None:
        x_bounds = [min(-2, np.min(x_list)), max(2, np.max(x_list))]
        y_bounds = [min(-2, np.min(y_list)), max(2, np.max(y_list))]
    else:
        x_bounds = [-2, 2]
        y_bounds = [-2, 2]

    x_ax = np.linspace(x_bounds[0], x_bounds[1], n)
    y_ax = np.linspace(y_bounds[0], y_bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = f(XX, YY)

    plt.figure()
    plt.contourf(XX, YY, ZZ, levels=50, cmap='jet')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y') 

    legend_elements = []    
    if x_list is not None and y_list is not None:
        plt.plot(x_list, y_list, color='purple', marker='o', linestyle='-', alpha=0.5)
        legend_elements.append('Gradient descent path')
        plt.scatter(x_list[-1], y_list[-1], color='yellow', marker='o', s=100)
        legend_elements.append('Final point')

    if starting_point is not None:
        plt.scatter(starting_point[0], starting_point[1], color='red', marker='x', s=100)
        legend_elements.append('Starting point')

    if global_min is not None:
        plt.scatter(global_min[0], global_min[1], color='green', marker='*', s=100)
        legend_elements.append('Global minimum')

    if len(legend_elements) > 0:
        plt.legend(legend_elements)

    plt.title('2D contour plot')
    plt.tight_layout()

    plt.savefig(f'plots/{figname}.pdf')
    plt.show()


def plot_function_over_iterations(f_list: np.ndarray, 
                                 figname: str = 'function_over_iterations') -> None:
    """
    Plot the function value over iterations.
    :param f_list: The list of function values
    :param figname: Filename to save the plot in the `plots` directory
    :return:
    """
    plt.plot(f_list)
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.title('Function value over iterations')
    plt.tight_layout()
    plt.savefig(f'plots/{figname}.pdf')
    plt.show()