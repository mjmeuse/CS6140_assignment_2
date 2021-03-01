import pandas as pd
import numpy as np
from numpy.linalg import inv
from math import log, pow, e, pi
from matplotlib import pyplot as plt
from copy import copy
import random
from scipy.stats import multivariate_normal


def confusion_matrix(actual_list, predicted_list):
    """
    Creates a K by K confusion matrix where K is the number of unique classes in the list of actual target classes.
    :param actual_list: list of the actual target class values.
    :param predicted_list: list of the predicted target class values.
    :return: a K by K confusion matrix.
    """
    class_to_index_map = dict([(y, x) for x, y in enumerate(sorted(set(actual_list)))])
    num_classes = len(np.unique(actual_list))
    matrix = np.zeros((num_classes, num_classes))
    for act, pred in zip(actual_list, predicted_list):
        matrix[class_to_index_map[act]][class_to_index_map[pred]] += 1
    return matrix


def import_data_set(path, labels=None, skip_rows=None):
    """
    Import CSV file using pandas.
    :param skip_rows: Number of rows to skip during import
    :param path: A string, the file path
    :param labels:
    :return:
    """
    return pd.read_csv(path, header=None, names=labels, skiprows=skip_rows)


def permute_data(data):
    """
    Generate random permutation of the data set.
    :param data: a matrix of the entire data set
    :return: the permutation
    """
    return data.iloc[np.random.permutation(len(data))].reset_index(drop=True)


def average_accuracy(actual, predicted):
    acc_list = []
    for act, pred in zip(actual, predicted):
        acc_list.append((np.abs(act - pred) / act))
    return sum(acc_list)/len(acc_list)



def accuracy(predicted_values, actual_values):
    """
    Calculate the accuracy between the predicted classification and the actual classification
    :param predicted_values: list of predicted classifications.
    :param actual_values: list of actual classifications.
    :return: float, the accuracy
    """
    count = 0
    if len(predicted_values) != len(actual_values):
        raise ValueError("Number of Predicted values and number of Actual values are not equal")
    for i in range(len(predicted_values)):
        if predicted_values[i] == actual_values[i]:
            count += 1
    return count / len(predicted_values)


def K_folds(dataframe, folds=10):
    """
    Generate K number of folds. By default K = 10. A fold is a subset of the dataset where
    fold size = Number of samples / number of folds rounded to the nearest whole number.
    Intended use case:
        K-1 folds will be used as training data, and 1 fold will be used as validation data.
        This process is repeated K times such that each fold is used as the validation set exactly one time.
    :param dataframe: the randomized dataset from which to create the K subsets.
    :param folds: The number of subsets to create.
    :return: A list of K subsets.
    """
    split_points = list(map(lambda x: int(x * len(dataframe) / folds), (list(range(1, folds)))))
    folds = list(np.split(dataframe.sample(frac=1).reset_index(drop=True), split_points))
    train, test = _process_Kfolds(folds)
    return train, test


def _process_Kfolds(folds):
    """
    Iterate through the K-folds and create the training data and validation data. K-1 folds will be used as training
    data, and 1 fold will be used as validation data. This process is repeated K times such that each fold is used
    as the validation set exactly one time.
    :param folds: list of K-folds
    :return: list of training sets, list of validation sets
    """
    test = []
    train = []
    for i in range(len(folds)):
        test.append(folds[i])
        left = folds[:i]
        right = folds[i + 1:]
        train.append(pd.concat(left + right))
    return train, test


def get_most_frequent(data):
    """
    Returns the class with highest frequency of occurrence during the recursion.
    :param data: The data to evaluate.
    :return: The class with the highest frequency.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    list_of_classes, frequency = np.unique(data[:, -1], return_counts=True)
    index = frequency.argmax()
    return list_of_classes[index]


def entropy(data):
    """
    Calculate the entropy
    :param data: data for which to calculate the entropy
    :return: float, the entropy
    """
    if len(data) == 0:
        return 0

    _entropy = 0.0
    frequency_dict = {}
    data = np.array(data)
    class_list, counts = np.unique(data[:, -1], return_counts=True)
    for class_value, count in zip(class_list, counts):
        frequency_dict[class_value] = count

    for count in frequency_dict.values():
        _entropy += (-count / len(data)) * log(count / len(data), 2)

    return _entropy


def information_gain(parent_set, subsets):
    """
    Calculate the information gained from splitting the parent on an attribute value.
    :param parent_set: The complete set of data which is currently being evaluated.
    :param subsets: The subsets of the parent_set post split.
    :return: (floa) the information gained from the split.
    """

    S = len(parent_set)

    impurityBeforeSplit = entropy(parent_set)

    weights = [len(subset) / S for subset in subsets]
    impurityAfterSplit = 0
    for i in range(len(subsets)):
        impurityAfterSplit += weights[i] * entropy(subsets[i])

    totalGain = impurityBeforeSplit - impurityAfterSplit
    return totalGain


def sum_squared_error(actual_values, predicted_value):
    """
    Calculate the sum of the errors of the model.
    :param predicted_value: predicted value, the mean of the actual values
    :param actual_values: list of actual values (also could be referred to as the target values)
    :return: float, the sum of the average_accuracy
    """
    return sum(np.square(np.subtract(actual_values, predicted_value)))


def mean_square_error(actual_values, predicted_values):
    return sum_squared_error(actual_values, predicted_values) / len(actual_values)


def overall_squared_error(data_above, data_below):
    """
    Calculate the mean squared average_accuracy of the result subsets after the data has been split.
    :param data_above: the subset of data above the split point
    :param data_below: the subset of data below the split point
    :return: (float) the resulting mean squared average_accuracy weighted by probability.
    """
    num_total_elements = len(data_below) + len(data_above)
    probability_data_above = len(data_above) / num_total_elements
    probability_data_below = len(data_below) / num_total_elements

    actual_values_data_above = data_above[:, -1]
    predicted_values_data_above = np.mean(actual_values_data_above)

    actual_values_data_below = data_below[:, -1]
    predicted_data_below = np.mean(actual_values_data_below)

    return (probability_data_above * sum_squared_error(actual_values_data_above, predicted_values_data_above) +
            probability_data_below * sum_squared_error(actual_values_data_below, predicted_data_below))


def mse(actual_values, predicted_values):
    """
    Calculate the root mean squared average_accuracy.
    :param actual_values: list of actual values
    :param predicted_values: list of predicted values
    :return: the root mean squared average_accuracy
    """
    error = 0
    for i in range(len(actual_values)):
        error += np.square(actual_values[i] - predicted_values[i])
    return error / len(actual_values)


def gradient_descent(data, target, learning_rate, tolerance, max_iterations=1000):
    rmse_list = []
    n, m = data.shape
    w = np.zeros(m)
    iteration = 0
    cost_delta = 1
    cur_cost = 0

    while cost_delta > tolerance and iteration < max_iterations:
        prev_cost = cur_cost
        cur_cost, sse = compute_rmse(data, target, w)
        for x, y in zip(data, target):
            error = np.dot(w, x) - y
            w = w - learning_rate * x * error
        rmse_list.extend([iteration, cur_cost])
        cost_delta = abs(cur_cost - prev_cost)
        iteration += 1
    return w, rmse_list


def compute_rmse(test_X, test_Y, theta):
    m = test_Y.size
    predict = test_X.dot(theta)
    error = predict - test_Y
    sse = np.sum(np.square(error) / m)
    rmse = np.sqrt(sse)
    return rmse, sse


def Zscore_normalization(matrix, input_mean_list=None, input_std_list=None):
    if input_mean_list is None and input_std_list is None:
        output_std_list = []
        output_mean_list = []
        norm_matrix = np.apply_along_axis(lambda x: ((x - np.mean(x)) / np.std(x)), 0, matrix)
        for i in range(matrix.shape[1]):
            output_std_list.append(np.std(matrix[:, i]))
            output_mean_list.append(np.mean(matrix[:, i]))
        return norm_matrix, output_std_list, output_mean_list
    elif input_mean_list is not None and input_std_list is not None:
        norm_matrix = None  # np.empty(shape=matrix.shape)
        for i in range(matrix.shape[1]):
            mean_val = input_mean_list[i]
            std_val = input_std_list[i]
            norm_matrix = np.apply_along_axis(lambda x: ((x - mean_val) / std_val), 0, matrix)
        return norm_matrix
    else:
        raise ValueError("Mean and Standard deviation must both be None or must both be not None")


def plotRMSE(iterations, RMSE_list, name, xlabel=None, ylabel=None):
    plt.plot(iterations, RMSE_list)
    plt.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(name)
    plt.show()
    plt.savefig(fname=name)


def ridge_regression(x, y, _lambda_):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    n, m = x.shape
    z = inv(np.dot(x.transpose(), x) + np.identity(m) * _lambda_)
    w = np.dot(np.dot(z, x.transpose()), y)
    return w


def add_ones(matrix):
    ones = np.ones((len(matrix), 1))
    with_ones = np.column_stack((ones, matrix))
    return with_ones


def center_data(df):
    columns = df.columns
    for column in columns:
        mean = np.mean(df[column])
        df[column] = df[column].apply(lambda x: x - mean)
    return df


def add_features(df, num_features):
    df_new = copy(df)
    for i in range(2, 2 + num_features):
        feature_column = df_new[df_new.columns[0]].apply(lambda x: x ** i)
        df_new[f"new_feature_{i}"] = feature_column
    return df_new


def generate_lamda_values(range_val):
    lamda_list = []
    i = 0.0
    while i < range_val:
        lamda_list.append(i)
        i += 0.2
    return lamda_list


def separate_target_from_data(matrix):
    x = None
    y = None
    if isinstance(matrix, pd.DataFrame):
        x = matrix.iloc[:, :-1]
        y = matrix.iloc[:, -1:]
    elif isinstance(matrix, np.ndarray):
        x = matrix[:, :-1]
        y = matrix[:, -1:]
    return x, y


def generate_data_sets(num_samples, mu1_list, sigma1_list, mu2_list, sigma2_list, w1_list, w2_list):
    assert len(mu1_list) == len(sigma1_list) == len(mu2_list) == len(sigma2_list), ValueError("Lengths of input lists "
                                                                                              "not equal")
    data_set_list = []
    for i in range(len(mu1_list)):
        data_set_list.append(generate_distribution(num_samples, mu1_list[i], sigma1_list[i], mu2_list[i],
                                                   sigma2_list[i], w1_list[i], w2_list[i]))
    return data_set_list


def generate_distribution(n, mu1, sigma1, mu2, sigma2, w1, w2):
    dist = []
    for i in range(round(n * w1)):
        dist.append(random.gauss(mu1, sigma1))
    for j in range(round(n * w2)):
        dist.append(random.gauss(mu2, sigma2))
    dist = np.random.permutation(dist)
    return dist


def probability_density_function(x, mu, sigma):
    pdf_list = []
    for val in x:
        pdf_list.append(1.0 / ((np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-0.5 * ((val - mu) / sigma) ** 2)))
    return np.array(pdf_list)


def e_step(data, mu, sigma, prior, num_distributions):
    n = data.shape[0]
    W = np.zeros((num_distributions, n), dtype=float)
    for k in range(num_distributions):
        W[k] = (multivariate_normal.pdf(data, mean=mu[k], cov=sigma[k], allow_singular=True)) * prior[k]
    Wsum = W.sum(axis=0)
    W = W / Wsum
    return W


def m_step(data, W, mu, sigma, nclust):
    n = data.shape[0]
    for t in range(nclust):
        temp1 = (data - mu[t]) * W.T[:, t][:, np.newaxis]
        temp2 = (data - mu[t]).T
        sigma[t] = np.dot(temp2, temp1) / W.sum(axis=1)[t]
        mu[t] = (data * W.T[:, t][:, np.newaxis]).sum(axis=0) / W.sum(axis=1)[t]
    prior = W.sum(axis=1) / n
    return mu, sigma, prior


def em_clustering(data, nclust, maxiter, init_mus, tolerance):
    # Initialization of mean, covariance, and prior
    n, d = data.shape
    mu = np.zeros((nclust, d), dtype=float)
    sigma = np.zeros((nclust, d, d), dtype=float)
    for t in range(nclust):  # assigning  data points to the means
        mu[t] = init_mus[t]
        sigma[t] = np.identity(d)
    prior = np.asarray(np.repeat(1.0 / nclust, nclust), dtype=float)  # for each cluster one prior:

    for i in range(maxiter):
        mu_old = 1 * mu
        W = e_step(data, mu, sigma, prior, nclust)  # calling E-step funct.
        mu, sigma, prior = m_step(data, W, mu, sigma, nclust)  # calling M-step funct.
        # checking stopping criterion
        rmse = 0
        for j in range(nclust):
            rmse = rmse + np.sqrt(np.power((mu[j] - mu_old[j]), 2).sum())
        rmse = round(rmse, 4)
        if rmse <= tolerance:
            break
    return mu.squeeze(), sigma.squeeze(), prior


def calculate_mu_sigma(x, w1, mu1, sigma1, w2, mu2, sigma2, learning_rate, n_iter=200):
    """
    Gradient descent algorithm for calculating mu, sigma, and priors for a gaussian mixture of two distributions
    :param x: total data
    :param w1: prior 1
    :param mu1: initial mean of distribution 1
    :param sigma1: initial standard deviation of distribution 1
    :param w2: prior 2
    :param mu2: initial mean of distribution 2
    :param sigma2: initial standard deviation of distribution 2
    :param learning_rate: hyper parameter, the step size along the gradient
    :param n_iter: max number of iterations
    :return: mu1 (float), sigma1 (float), mu2 (float), sigma2 (float), w1 (float), w2 (float)
    """
    mu1_list = [mu1]
    sigma1_list = [sigma1]
    mu2_list = [mu2]
    sigma2_list = [sigma2]
    calc_w1 = w1
    calc_w2 = w2

    x = np.array(x)

    for i in range(n_iter):
        d1 = []
        d2 = []

        p_dist_1 = probability_density_function(x, mu1, sigma1) * calc_w1
        p_dist_2 = probability_density_function(x, mu2, sigma2) * calc_w2

        mu_grad1 = (np.mean(x) - mu1) / (sigma1 ** 2)
        sigma_grad1 = np.mean((x - mu1) ** 2) / (sigma1 ** 3) - 1 / sigma1
        mu1 = mu1 + learning_rate * mu_grad1
        sigma1 = sigma1 + learning_rate * sigma_grad1
        mu1_list.append(mu1)
        sigma1_list.append(sigma1)

        mu_grad2 = (np.mean(x) - mu2) / (sigma2 ** 2)
        sigma_grad2 = np.mean((x - mu2) ** 2) / (sigma2 ** 3) - 1 / sigma2
        mu2 = mu2 + learning_rate * mu_grad2
        sigma2 = sigma2 + learning_rate * sigma_grad2
        mu2_list.append(mu2)
        sigma2_list.append(sigma2)

        for k in range(len(x)):
            temp1 = p_dist_1[k] / (p_dist_1[k] + p_dist_2[k])
            temp2 = p_dist_2[k] / (p_dist_1[k] + p_dist_2[k])
            if temp1 > temp2:
                d1.append(p_dist_1[k])
            elif temp1 < temp2:
                d2.append(p_dist_2[k])
            else:
                rand_val = random.randrange(0, 2, 1)
                if rand_val > 0:
                    d1.append(p_dist_1[k])
                else:
                    d2.append(p_dist_2[k])

        calc_w1 = len(d1) / len(x)
        calc_w2 = len(d2) / len(x)
        # print(f"w1 = {w1}")
        # print(f"w2 = {w2}")

    return mu1_list[-1], sigma1_list[-1], mu2_list[-1], sigma2_list[-1], calc_w1, calc_w2


def get_mu_sigma(x, mu, sigma, learning_rate, n_iter=1000):
    mu_list = [mu]
    sigma_list = [sigma]
    x = np.array(x)
    for i in range(n_iter):
        mu_grad = (np.mean(x) - mu) / (sigma ** 2)
        sigma_grad = np.mean((x - mu) ** 2) / (sigma ** 3) - 1 / sigma
        mu = mu + learning_rate * mu_grad
        sigma = sigma + learning_rate * sigma_grad
        mu_list.append(mu)
        sigma_list.append(sigma)
    return mu_list[-1], sigma_list[-1], mu_list, sigma_list

# def _update(x, mu, sigma, learning_rate):
#     mu_grad = (np.mean(x) - mu) / (sigma ** 2)
#     sigma_grad = np.mean((x - mu) ** 2) / (sigma ** 3) - 1 / sigma
#     mu = mu + learning_rate * mu_grad
#     sigma = sigma + learning_rate * sigma_grad
#     return mu_grad, sigma_grad, mu, sigma
