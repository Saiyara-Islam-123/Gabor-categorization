import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_gabor_data  # Importing the data loading function from dataset.py
from Net import Net, SupervisedNet  # Import the autoencoder and supervised model from Net.py
import matplotlib.pyplot as plt
import os
import numpy as np
from dist import *
import pandas as pd
import itertools

######################for XAB pairs or for all pairs, frequency training ############################################
def train_supervised(model, trainloader, device, lr, weights_dir, csv_dir, epochs=15, dist_func = sampled_all_distance):

    # Define the loss function specific for supervised learning
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model.train()

    # Create the folder for saving results if it doesn't exist


    loss_values = []
    accuracy_values = []

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []


    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        batch = 0
        for images, labels in trainloader:
            # Prepare the images and labels
            images = images.to(device)  # Move input images to the same device as the model

            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)


            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            encoder_outputs = model.encoder_output
            zero, zero_one, one = dist_func(encoder_outputs, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accuracy_values.append(accuracy)
            print(accuracy)

            #weights_dir = "../net_weights/sup"
            #os.makedirs(weights_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist
            torch.save(model.state_dict(), f"../net_weights/Prev runs/{weights_dir}/sup_net_weights_ lr={lr} "+str(epoch)+  " " + str(batch) +".pth")
            print(zero, zero_one, one)
            batch += 1

        # Compute average loss and accuracy for the epoch
            avg_loss = running_loss / len(trainloader)
            loss_values.append(avg_loss)

            print(f"Supervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f},")





    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]
    df["acc"] = accuracy_values
    df.to_csv(f"Prev runs/{csv_dir}/LR={lr}, Distance every batch sup, epochs.csv", index=False)




##############################################################################################################################