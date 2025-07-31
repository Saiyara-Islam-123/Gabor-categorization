import random
import torch.nn.functional
from dataset import load_gabor_data
random.seed(0)
import numpy as np
import pandas as pd

def cos(a, b):
    dot_product = torch.dot(a, b)
    return dot_product / (torch.norm(a) * torch.norm(b))

def xab_pairs_dist(X, y):
    d = {}
    d[(0, 0)] = []
    d[(0, 1)] = []
    d[(1, 1)] = []
    for i in range(0, y.shape[0], 2):
        y1 = y[i].item()
        y2 = y[i+1].item()
        mat_1 = X[i]
        mat_2 = X[i+1]

        mat_1_flattened = mat_1.view(mat_1.size(0), -1)
        mat_2_flattened = mat_2.view(mat_2.size(0), -1)


        if y1 == 1 and y2 == 0:

            d[(y2, y1)].append(1-cos(mat_1_flattened, mat_2_flattened).item())
        else:
            d[(y1, y2)].append(1-cos(mat_1_flattened, mat_2_flattened).item())

    within_zero, between, within_one = d[(0, 0)], d[(0, 1)], d[(1, 1)]

    return np.mean(np.array(within_zero)), np.mean(np.array(between)), np.mean(np.array(within_one))



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


                mat_1_flattened = mat_1.view(mat_1.size(0), -1).reshape(mat_1.size(0))
                mat_2_flattened = mat_2.view(mat_2.size(0), -1).reshape(mat_1.size(0))


                if y1 == 1 and y2 == 0:

                    d[(y2, y1)].append(1-cos(mat_1_flattened, mat_2_flattened).item())
                else:
                    d[(y1, y2)].append(1-cos(mat_1_flattened, mat_2_flattened).item())


    within_zero, between, within_one =  d[(0,0)], d[(0,1)], d[(1,1)]


    return np.mean(np.array(within_zero)), np.mean(np.array(between)), np.mean(np.array(within_one))



#I basically find the Euclidean distance between two random datapoints from these two bigger matrices.

if __name__== '__main__':
    excel_file = "categorisation 4000.xlsx"

    train_loader, _, _ = load_gabor_data(excel_file, batch_size=32)
    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []
    df = pd.DataFrame()


    for images, labels in train_loader:

        zero, zero_one, one = (sampled_all_distance(X=images.reshape(32, 3*128*128), y=labels.reshape(32, 1)))
        avg_distances[(0, 0)].append(zero)
        avg_distances[(0, 1)].append(zero_one)
        avg_distances[(1, 1)].append(one)
        break

    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]
    df.to_csv("Distance no train.csv", index=False)
    
