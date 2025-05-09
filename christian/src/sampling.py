import random
import torch.nn.functional

random.seed(0)
import numpy as np

def sampled_all_distance(X,y):
    d = {}
    d[(0,0)] = []
    d[(0,1)] = []
    d[(1,1)] = []

    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            if i != j:
                y1 = y[i].item()
                y2 = y[j].item()

                mat_1 = X[i]
                mat_2 = X[j]

                mat_1_flattened = mat_1.view(mat_1.size(0), -1)
                mat_2_flattened = mat_2.view(mat_2.size(0), -1)

                mat_1_flattened_normalized = torch.nn.functional.normalize(mat_1_flattened, p=2, dim=1)
                mat_2_flattened_normalized = torch.nn.functional.normalize(mat_2_flattened, p=2, dim=1)

                if y1 == 1 and y2 == 0:

                    d[(y2, y1)].append(torch.norm(mat_1_flattened_normalized - mat_2_flattened_normalized).item())
                else:
                    d[(y1, y2)].append(torch.norm(mat_1_flattened_normalized - mat_2_flattened_normalized).item())


    within_zero, between, within_one =  d[(0,0)], d[(0,1)], d[(1,1)]

    return np.mean(np.array(within_zero)), np.mean(np.array(between)), np.mean(np.array(within_one))


#I basically find the Euclidean distance between two random datapoints from these two bigger matrices.

