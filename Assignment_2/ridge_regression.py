from helper_functions import K_folds, import_data_set, center_data, add_features, generate_lamda_values, \
    ridge_regression, separate_target_from_data, plotRMSE, compute_rmse
import numpy as np


def main():
    data_set_list = []
    raw_data = import_data_set("../Datasets/sinData_Train.csv")
    raw_data = center_data(raw_data)
    data_set_list.append(add_features(raw_data, 4))
    data_set_list.append(add_features(raw_data, 8))

    for data_set in data_set_list:
        training_sets, validations_sets = K_folds(data_set)
        lambda_vals = generate_lamda_values(10)
        train_average_rmse_list = []
        test_average_rmse_list = []
        rmse_test_all_folds_list = []
        rmse_train_all_folds_list = []
        for training_set, validations_set in zip(training_sets, validations_sets):
            train_rmse_list = []
            test_rmse_list = []
            train_x, train_y = separate_target_from_data(training_set)
            test_x, test_y = separate_target_from_data(validations_set)
            for lamda_value in lambda_vals:
                w_vector = ridge_regression(train_x, train_y, lamda_value)
                rmse_train, sse_train = compute_rmse(train_x.to_numpy(), train_y.to_numpy(), w_vector)
                train_rmse_list.append(rmse_train)
                rmse_test, sse_test = compute_rmse(test_x.to_numpy(), test_y.to_numpy(), w_vector)
                test_rmse_list.append(rmse_test)

            rmse_test_all_folds_list.append(train_rmse_list)
            rmse_train_all_folds_list.append(test_rmse_list)

        for i in range(len(lambda_vals)):
            train_rmse_lambda = np.array(rmse_test_all_folds_list)[:, i]
            train_average_rmse_list.append(sum(train_rmse_lambda) / len(rmse_test_all_folds_list))
            train_rmse_lambda = np.array(rmse_test_all_folds_list)[:, i]
            test_average_rmse_list.append(sum(train_rmse_lambda) / len(rmse_test_all_folds_list))
        plotRMSE(lambda_vals, train_average_rmse_list, f"traning_avg_rmse with {len(data_set.columns)} features", xlabel="Lambda", ylabel="AVG RMSE")
        plotRMSE(lambda_vals, test_average_rmse_list, f"testing_avg_rmse {len(data_set.columns)} features", xlabel="Lambda", ylabel="AVG RMSE")




if __name__ == '__main__':
    main()
