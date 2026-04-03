import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations, pairplot)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import rastrigin, gradient_rastrigin, gradient_descent


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, 
                    "duration": 4, "exercise_intensity": 5, 
                    "fitness_level": 6, "calories": 7}

    # After loading the data, you can for example access it like this: 
    # `smartwatch_data[:, column_to_id['hours_sleep']]`
    smartwatch_data = np.load('data/smartwatch_data.npy') ##  Load the smartwatch data from a .npy file 

    pairplot(smartwatch_data, list(column_to_id.keys())) ##  Plot pairplot to visualize relationships between features

    # TODO: Implement Task 1.1.1: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.

    # 3 Linearly Dependent Pairs
    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["calories"]]
    if use_linalg_formulation:
        X = compute_design_matrix(x.reshape(-1, 1))
        theta = fit_multiple_lin_model(X, y)
        loss = multiple_loss(X, y, theta)
    else:
        theta = fit_univariate_lin_model(x, y)
        loss = univariate_loss(x, y, theta)
    prs_corr = calculate_pearson_correlation(x, y)
    print(f"Mean-squared error (MSE) as the loss function is: {loss}.")
    print(f"Pearson correlation for 'duration' and 'calories' features is: {prs_corr}.")
    print(f"Theta vector is equalt to: {theta}.")
    plot_scatterplot_and_line(x, y, theta, "Duration", "Calories", "Duration vs Calories")

    x = smartwatch_data[:, column_to_id["avg_pulse"]]
    y = smartwatch_data[:, column_to_id["max_pulse"]]
    if use_linalg_formulation:
        X = compute_design_matrix(x.reshape(-1, 1))
        theta = fit_multiple_lin_model(X, y)
        loss = multiple_loss(X, y, theta)
    else:
        theta = fit_univariate_lin_model(x, y)
        loss = univariate_loss(x, y, theta)
    prs_corr = calculate_pearson_correlation(x, y)
    print(f"Mean-squared error (MSE) as the loss function is: {loss}.")
    print(f"Pearson correlation for 'avg_pulse' and 'max_pulse' features is: {prs_corr}.")
    print(f"Theta vector is equalt to: {theta}.")
    plot_scatterplot_and_line(x, y, theta, "Avg_pulse", "Max_pulse", "Avg_pulse vs Max_pulse")

    x = smartwatch_data[:, column_to_id["duration"]]
    y = smartwatch_data[:, column_to_id["fitness_level"]]
    if use_linalg_formulation:
        X = compute_design_matrix(x.reshape(-1, 1))
        theta = fit_multiple_lin_model(X, y)
        loss = multiple_loss(X, y, theta)
    else:
        theta = fit_univariate_lin_model(x, y)
        loss = univariate_loss(x, y, theta)
    prs_corr = calculate_pearson_correlation(x, y)
    print(f"Mean-squared error (MSE) as the loss function is: {loss}.")
    print(f"Pearson correlation for 'duration' and 'fitness_level' features is: {prs_corr}.")
    print(f"Theta vector is equalt to: {theta}.")
    plot_scatterplot_and_line(x, y, theta, "Duration", "Fitness_level", "Duration vs Fitness_level")

    # 3 Not-Linearly Dependent Pairs
    x = smartwatch_data[:, column_to_id["hours_sleep"]]
    y = smartwatch_data[:, column_to_id["fitness_level"]]
    if use_linalg_formulation:
        X = compute_design_matrix(x.reshape(-1, 1))
        theta = fit_multiple_lin_model(X, y)
        loss = multiple_loss(X, y, theta)
    else:
        theta = fit_univariate_lin_model(x, y)
        loss = univariate_loss(x, y, theta)
    prs_corr = calculate_pearson_correlation(x, y)
    print(f"Mean-squared error (MSE) as the loss function is: {loss}.")
    print(f"Pearson correlation for 'hours_sleep' and 'fitness_level' features is: {prs_corr}.")
    print(f"Theta vector is equalt to: {theta}.")
    plot_scatterplot_and_line(x, y, theta, "Hours_sleep", "Fitness_level", "Hours_sleep vs Fitness_level")

    x = smartwatch_data[:, column_to_id["hours_work"]]
    y = smartwatch_data[:, column_to_id["exercise_intensity"]]
    if use_linalg_formulation:
        X = compute_design_matrix(x.reshape(-1, 1))
        theta = fit_multiple_lin_model(X, y)
        loss = multiple_loss(X, y, theta)
    else:
        theta = fit_univariate_lin_model(x, y)
        loss = univariate_loss(x, y, theta)
    prs_corr = calculate_pearson_correlation(x, y)
    print(f"Mean-squared error (MSE) as the loss function is: {loss}.")
    print(f"Pearson correlation for 'hours_work' and 'exercise_intensity' features is: {prs_corr}.")
    print(f"Theta vector is equalt to: {theta}.")
    plot_scatterplot_and_line(x, y, theta, "Hours_work", "Exercise_intensity", "Hours_work vs Exercise_intensity")

    x = smartwatch_data[:, column_to_id["avg_pulse"]]
    y = smartwatch_data[:, column_to_id["duration"]]
    if use_linalg_formulation:
        X = compute_design_matrix(x.reshape(-1, 1))
        theta = fit_multiple_lin_model(X, y)
        loss = multiple_loss(X, y, theta)
    else:
        theta = fit_univariate_lin_model(x, y)
        loss = univariate_loss(x, y, theta)
    prs_corr = calculate_pearson_correlation(x, y)
    print(f"Mean-squared error (MSE) as the loss function is: {loss}.")
    print(f"Pearson correlation for 'avg_pulse' and 'duration' features is: {prs_corr}.")
    print(f"Theta vector is equalt to: {theta}.")
    plot_scatterplot_and_line(x, y, theta, "Avg_pulse", "Duration", "Avg_pulse vs Duration")       

    # TODO: Implement Task 1.2.2: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.
    x_M = smartwatch_data[:, [column_to_id["duration"], column_to_id["exercise_intensity"], column_to_id["avg_pulse"]]]
    y_M = smartwatch_data[:, column_to_id["calories"]]

    X_M = compute_design_matrix(x_M)
    theta_M = fit_multiple_lin_model(X_M, y_M)
    loss_M = multiple_loss(X_M, y_M, theta_M)

    print(f"Theta_M: {theta_M}")
    print(f"MSE_M: {loss_M}")

    x_U = smartwatch_data[:, column_to_id["duration"]]
    y_U = smartwatch_data[:, column_to_id["calories"]]

    theta_U = fit_univariate_lin_model(x_U, y_U)
    loss_U = univariate_loss(x_U, y_U, theta_U)

    print(f"Theta_U: {theta_U}")
    print(f"MSE_U: {loss_U}")

    # TODO: Implement Task 1.3.1: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    x_poly = smartwatch_data[:, column_to_id["duration"]]
    y_poly = smartwatch_data[:, column_to_id["calories"]]
    
    #K = 2
    #K = 3
    K = 4

    X_poly = compute_polynomial_design_matrix(x_poly, K)
    theta_poly = fit_multiple_lin_model(X_poly, y_poly)
    loss_poly = multiple_loss(X_poly, y_poly, theta_poly)

    print(f"Theta poly is: {theta_poly}.")
    print(f"Mean-squared error (MSE) for polynomial regression is: {loss_poly}.")
    plot_scatterplot_and_polynomial(x_poly, y_poly, theta_poly, "Duration", "Calories", "Duration vs Calories (Polynomial)")



def task_2():
    print('\n---- Task 2 ----')

    for task in [1,2,3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-1.npy')
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load('data/X-1-data.npy')
            y = np.load('data/targets-dataset-2.npy')
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load('data/X-2-data.npy')
            y = np.load('data/targets-dataset-3.npy')
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # TODO: Split the dataset using the `train_test_split` function.
        # The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # TODO: Fit the model to the data using the `fit` method of the classifier `clf`
        clf.fit(X_train, y_train)

        # TODO: Use the `score` method of the classifier `clf` to calculate accuracy
        acc_train = clf.score(X_train, y_train)
        acc_test = clf.score(X_test, y_test)

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        # TODO: Use the `predict_proba` method of the classifier `clf` to
        #  calculate the predicted probabilities on the training set
        yhat_train = clf.predict_proba(X_train)
        # TODO: Use the `predict_proba` method of the classifier `clf` to
        #  calculate the predicted probabilities on the test set
        yhat_test = clf.predict_proba(X_test) 

        # TODO: Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train = log_loss(y_train, yhat_train)
        loss_test = log_loss(y_test, yhat_test)

        print(f'Train loss: {loss_train}. Test loss: {loss_test}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        # TODO: Print theta vector (and also the bias term). Hint: Check the attributes of the classifier
        classifier_weights = clf.coef_
        classifier_bias = clf.intercept_
        print(f'Parameters: {classifier_weights}, {classifier_bias}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(46)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = np.array(np.random.randn())
    y0 = np.array(np.random.randn())
    print(f'Starting point: {x0:.4f}, {y0:.4f}')

    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(rastrigin)
        plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0))   
    
    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.

    # Find optimal learning rate and learning decay
    # -----------------------------------------------------------------------------------------------------------------
    # for lrng_rate in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]:
    #     for lrng_decay in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
         
    #         x_list, y_list, f_list = gradient_descent(rastrigin, gradient_rastrigin, x0, y0, learning_rate=lrng_rate,
    #                                                   lr_decay=lrng_decay, num_iters=100)
        
    #         print(f"Learning rate = {lrng_rate}, decay = {lrng_decay}")
    #         print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    # -----------------------------------------------------------------------------------------------------------------    


    x_list, y_list, f_list = gradient_descent(rastrigin, gradient_rastrigin, x0, y0, learning_rate=0.02,
                                              lr_decay=0.6, num_iters=100)
    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {rastrigin(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    plot_function_over_iterations(f_list)
    

def main():
    np.random.seed(46)

    task_1(use_linalg_formulation=True)
    task_2()
    task_3(initial_plot=True)

if __name__ == '__main__':
    main()

