import pandas as pd
import cvxopt.modeling as cm
from helper_functions import mean_square_error



def lin_regression():

    raw_data = pd.read_csv("../Datasets/winequality-red.csv", sep=";", header=0)
    raw_training = raw_data[:1500]
    x = cm.matrix(raw_training.iloc[:, :-1].to_numpy(dtype=float))
    y = cm.matrix(raw_training.iloc[:, -1:].to_numpy(dtype=float))

    raw_test = raw_data[1500:].reset_index(drop=True)
    x_test = cm.matrix(raw_test.iloc[:, :-1].to_numpy(dtype=float))
    y_test = cm.matrix(raw_test.iloc[:, -1:].to_numpy(dtype=float))

    a = cm.variable(x.size[1])
    b = cm.variable()
    z = cm.variable(x.size[0])

    constraint_1 = (z >= (y - x * a - b))
    constraint_2 = (z >= (x * a + b - y))

    z_min = cm.op(cm.min(cm.sum(z) / x.size[0]), [constraint_1, constraint_2])
    z_min.solve()

    calc_a = a.value
    calc_b = b.value

    z_train = y - x * calc_a - calc_b
    z_test = y_test - x_test * calc_a - calc_b

    train_results = x * calc_a + calc_b
    average_training_error = mean_square_error(y, train_results)

    test_results = x_test * calc_a + calc_b
    average_test_error = mean_square_error(y_test, test_results)

    print(f"average training error = {average_training_error}")
    print(f"average testing error = {average_test_error}")







if __name__ == '__main__':
    lin_regression()
