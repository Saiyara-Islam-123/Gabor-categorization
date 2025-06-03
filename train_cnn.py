import neural_network
from neural_network import *
from torch import optim
from dataset import *
from sampling import *
import pandas as pd
from reconstruction import *
import numpy as np

trainloader, valloader, testloader = load_gabor_data(excel_file="categorisation.xlsx")

def unsup_training():

    unsup_model = AutoEncoder()
    loss_fn_unsup = nn.MSELoss()

    avg_distances = {}
    avg_distances[(0,0)] = []
    avg_distances[(0,1)] = []
    avg_distances[(1,1)] = []

    optimizer = optim.Adam(unsup_model.parameters(), lr=0.01, weight_decay=0.01)
    unsup_model.train()

    print("\nUnsupervised part!")
    for epoch in range(5):
        for images, labels in trainloader:
            optimizer.zero_grad()


            outputs = unsup_model(images)
            loss = loss_fn_unsup(outputs, images)

            zero, zero_one, one = sampled_all_distance(unsup_model.encoded, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)

            loss.backward()
            optimizer.step()

            print(loss.item(), epoch)

    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]
    df.to_csv("Distance per batch unsup non-learner.csv", index=False)


    return unsup_model


def sup_training(unsup_model):
    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []
    accuracy_values = []

    sup_model = LastLayer(unsup_model)

    sup_model.train()

    loss_fn_sup = nn.CrossEntropyLoss()
    optimizer = optim.Adam(sup_model.parameters(), lr=0.005, weight_decay=0.0)

    print("\nSupervised part!")
    for epoch in range(5):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = sup_model(images)

            loss = loss_fn_sup(outputs, labels)

            zero, zero_one, one = sampled_all_distance(sup_model.encoder_output, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)
            accuracy_values.append(acc(sup_model))
            loss.backward()
            optimizer.step()
            print(loss.item(), epoch)


    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]
    df["Accuracy"] = accuracy_values
    df.to_csv("Distance per batch sup non-learner.csv", index=False)

    return sup_model

def acc(sup_model): #percentage correct
    for images, labels in testloader:
        outputs = sup_model(images)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == labels).float().mean().item()
        print("Accuracy: " + str(acc * 100))
    return acc


if __name__ == "__main__":
    unsup_model = unsup_training()

    sup_model = sup_training(unsup_model)
