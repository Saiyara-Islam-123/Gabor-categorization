import random
import torch.nn.functional

random.seed(0)

def sampled_avg_distance(pair, X, y):

    X_first = []
    X_second = []

    for i in range(y.size(0)):
        row = y[i]
        if row in pair:
            if row == pair[0]:
                X_first.append(X[i])
            if row == pair[1]:
                X_second.append(X[i])

    #I want to have batch sizes of 20 then do average. Then average the averages
    distances = []
    averages = []
    d = {}
    for i in range(5):
        for j in range(5):
            distances = []
            index1 = random.randint(0, len(X_first) - 1) #44
            index2 = random.randint(0, len(X_second) - 1) #43

            pair = X_first[index1], X_second[index2]
            pair_inverse = X_second[index2], X_first[index1] #43 44

            while pair in d or pair_inverse in d or index1 == index2:
                index1 = random.randint(0, len(X_first) - 1)
                index2 = random.randint(0, len(X_second) - 1)
                pair = X_first[index1], X_second[index2]
                pair_inverse = X_second[index2], X_first[index1]


            d[pair] = True

            mat_1 = X_first[index1]
            mat_2 = X_second[index2]

            mat_1_flattened = mat_1.view(mat_1.size(0), -1)
            mat_2_flattened = mat_2.view(mat_2.size(0), -1)

            mat_1_flattened_normalized = torch.nn.functional.normalize(mat_1_flattened, p=2, dim=1)
            mat_2_flattened_normalized = torch.nn.functional.normalize(mat_2_flattened, p=2, dim=1)

            distances.append(torch.norm(mat_1_flattened_normalized - mat_2_flattened_normalized))
        averages.append(sum(distances)/len(distances))

    return (sum(averages) / len(averages)).tolist()


#I basically find the Euclidean distance between two random datapoints from these two bigger matrices.

