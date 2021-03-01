from helper_functions import permute_data, K_folds, import_data_set, gradient_descent, Zscore_normalization, \
    compute_rmse, plotRMSE, add_ones
import argparse
import numpy as np




def main(args):
    path = f'../Datasets/{args.data}.csv'
    tolerance = args.t
    learning_rate = args.l
    num_folds = args.k
    raw_data = import_data_set(path)
    shuffled_data = permute_data(raw_data)
    sse_list = []
    i = 1
    training_sets, validation_sets = K_folds(shuffled_data, num_folds)
    for training_set, validation_set in zip(training_sets, validation_sets):
        normalized_training_data, std, mean = Zscore_normalization(training_set.to_numpy())
        normalized_training_data = add_ones(normalized_training_data)
        train_x = normalized_training_data[:, :-1]
        train_y = normalized_training_data[:, -1:].squeeze()

        normalized_test_data = Zscore_normalization(validation_set.to_numpy(), mean, std)
        normalized_test_data = add_ones(normalized_test_data)
        test_x = normalized_test_data[:, :-1]
        test_y = normalized_test_data[:, -1:].squeeze()

        theta, rmse_list = gradient_descent(train_x, train_y, learning_rate, tolerance)
        rmse, sse = compute_rmse(test_x, test_y, theta)
        sse_list.append(sse)
        if i == 1:
            plotRMSE(rmse_list[::2], rmse_list[1::2], f"{args.data}")
        print(f"Statistics for Fold iteration {i}:")
        print(f"The training RMSE is: {rmse_list[-1]}")
        print(f"The test RMSE is: {rmse}")
        i += 1
    print(f"The average SSE across all folds is: {sum(sse_list) / len(sse_list)}")
    print(f"The standard deviation for the SSE is: {np.std(sse_list)}")


















if __name__ == '__main__':
    description = "Gradient Descent"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-data', type=str, help=f'The dataset to use.')
    parser.add_argument('-k', type=int, help=f'Number of K folds to use in cross validation. Default = 10',
                        default=10)
    parser.add_argument('-t', type=float, help='The tolerance to use, a float.')
    parser.add_argument('-l', type=float, help='The learning rate to use, a float.')

    user_args = parser.parse_args()
    main(user_args)
