import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer

from dataset import load_gabor_data  # Importing the data loading function from dataset.py
from Net import Net
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import numpy as np
from dist import *
import pandas as pd
import itertools
from Transformer import *

def train_unsupervised(model, trainloader_freq, device, lr, weights_dir, csv_dir, epochs=5, dist_func = sampled_all_distance):

    # Define the loss function specific for autoencoder
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    loss_values = []

    # Create the folder for saving results if it doesn't exist
    #results_dir = "../epochs_results"
    #os.makedirs(results_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist

    # Initialize the plot for real-time visualization
    #plt.ion()
    #fig, ax = plt.subplots()
    #ax.set_title("Training Loss Over Epochs")
    #ax.set_xlabel("Epoch")
    #ax.set_ylabel("Loss")
    #loss_line, = ax.plot([], [], label="Loss", color="blue")  # Create the line for loss
    #ax.legend()  # Add legend once
    avg_distances_freq = {}
    avg_distances_freq[(0,0)] = []
    avg_distances_freq[(0, 1)] = []
    avg_distances_freq[(1, 1)] = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch = 0
        for images_freq, labels_freq in trainloader_freq:
            images_freq = images_freq.to(device)  # Move input images to the same device as the model

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images_freq)


            loss = criterion(outputs, images_freq)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            zero_freq, zero_one_freq, one_freq = dist_func(model.encoded, labels_freq)
            print(zero_freq, zero_one_freq, one_freq)

            avg_distances_freq[(0, 0)].append(zero_freq)
            avg_distances_freq[(0, 1)].append(zero_one_freq)
            avg_distances_freq[(1, 1)].append(one_freq)

            avg_loss = running_loss / len(trainloader_freq)
            loss_values.append(avg_loss)


            torch.save(model.state_dict(), f"../net_weights/{weights_dir}/unsup_weights_" + " lr= " + str(lr) + " " +str(epoch)+ " " + str(batch) +".pth")
            batch += 1

            print(f"Unsupervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


    df = pd.DataFrame()
    df["within 0"] = avg_distances_freq[(0,0)]
    df["within 1"] = avg_distances_freq[(1,1)]
    df["between"] = avg_distances_freq[(0,1)]


    df.to_csv(f"{csv_dir}/LR={lr}, Transformer Distance every batch unsup.csv", index=False)


def unsup_trainer(weights_dir, csv_dir):
    excel_file = "categorisation 4000.xlsx"

    # Load the data
    trainloader, valloader, testloader = load_gabor_data(excel_file, batch_size=32)

    # Initialize the autoencoder model
    unsup_net = Net()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the model
    train_unsupervised(unsup_net, trainloader_freq=trainloader, device=device, lr=0.005, epochs=1,
                       dist_func=sampled_all_distance, weights_dir=weights_dir, csv_dir=csv_dir)

if __name__ == "__main__":
    # Path to your Excel file
    # Define the relative path
    excel_file = "categorisation 4000.xlsx"

    # Load the data
    trainloader,valloader, testloader = load_gabor_data(excel_file,batch_size=100)

    # Initialize the autoencoder model
    unsup_net = Transformer()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the model
    train_unsupervised(unsup_net, trainloader_freq=trainloader, device = device, lr=0.001, epochs=1, dist_func=sampled_all_distance, weights_dir="Transformer_unsup", csv_dir=".")

