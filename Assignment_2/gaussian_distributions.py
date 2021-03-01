from helper_functions import generate_data_sets, em_clustering, mean_square_error
import numpy as np
import random
from statistics import stdev



def main():
    true_mu1_list = []
    true_mu2_list = []
    true_sigma1_list = []
    true_sigma2_list = []
    true_w1_list = []
    true_w2_list = []
    for i in range(30):
        true_mu1_list.append(random.randrange(2, 5, 1))
        true_mu2_list.append(random.randrange(6, 13, 1))
        true_sigma1_list.append(random.randrange(1, 3, 1))
        true_sigma2_list.append(random.randrange(1, 4, 1))
        true_w1_list.append(random.randrange(10, 90, 10) / 100)
        true_w2_list.append(1-true_w1_list[i])

    data_sets_100 = generate_data_sets(100, true_mu1_list, true_sigma1_list, true_mu2_list, true_sigma2_list,
                                       true_w1_list, true_w2_list)
    data_sets_1000 = generate_data_sets(1000, true_mu1_list, true_sigma1_list, true_mu2_list, true_sigma2_list,
                                        true_w1_list, true_w2_list)
    data_sets_10000 = generate_data_sets(10000, true_mu1_list, true_sigma1_list, true_mu2_list, true_sigma2_list,
                                         true_w1_list, true_w2_list)
    all_data = [data_sets_100, data_sets_1000, data_sets_10000]
    for data_set_list in all_data:
        print(f"Stats for 30 data sets with {len(data_set_list[0])} samples:\n")
        mu1_list = []
        mu2_list = []
        sigma1_list = []
        sigma2_list = []
        w1_list = []
        w2_list = []
        for mixture in data_set_list:
            start_mus = []
            for j in range(2):
                start_mus.append(random.randrange(2, 13, 1))
            start_mus.sort()
            mus, sigmas, ws = em_clustering(np.array(mixture).reshape(len(mixture), 1), 2, 1500, start_mus, .001)
            mu1_list.append(mus[0])
            mu2_list.append(mus[1])
            sigma1_list.append(sigmas[0])
            sigma2_list.append(sigmas[1])
            w1_list.append(ws[0])
            w2_list.append(ws[1])

        mu1_acc = mean_square_error(true_mu1_list, mu1_list)
        mu1_std = stdev(mu1_list)

        mu2_acc = mean_square_error(true_mu2_list, mu2_list)
        mu2_std = stdev(mu2_list)

        sigma1_acc = mean_square_error(true_sigma1_list, sigma1_list)
        sigma1_std = stdev(sigma1_list)

        sigma2_acc = mean_square_error(true_sigma2_list, sigma2_list)
        sigma2_std = stdev(sigma2_list)

        w1_acc = mean_square_error(true_w1_list, w1_list)
        w1_std = stdev(w1_list)

        w2_acc = mean_square_error(true_w2_list, w2_list)
        w2_std = stdev(w2_list)

        print(f"mu1 mean squared error = {mu1_acc}, mean squared error mu2 = {mu2_acc}, mean squared error sigma1 = "
              f"{sigma1_acc}, mean squared error sigma2 = {sigma2_acc}, mean squared error w1 = {w1_acc}, "
              f"mean squared error w2 = {w2_acc}")

        print(f"mu1 stdev = {mu1_std}, mu2 stdev = {mu2_std}, sigma1 stdev = {sigma1_std}, "
              f"sigma2 stdev = {sigma2_std}, w1 stdev = {w1_std}, w2 stdev = {w2_std}\n")



if __name__ == "__main__":
    main()
