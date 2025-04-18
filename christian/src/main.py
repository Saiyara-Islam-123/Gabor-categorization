import torch

from train_unsupervised import train_unsupervised
from dataset import load_gabor_data  # Importing the data loading function from dataset.py
#from Net import Net,SupervisedNet  # Import the autoencoder model from Net.py
from Net import Net,SupervisedNet  # Import the autoencoder model from Net.py
from train_unsupervised import train_unsupervised
from train_supervised import train_supervised
import os
import numpy as np

if __name__ == "__main__":

    # Path to your Excel file
    # Define the relative path
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")

    # Load the data
    trainloader, valloader, testloader = load_gabor_data(excel_file,batch_size=64)


    # Initialize the autoencoder model
    unsup_net = Net()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the unsupervised model
    train_unsupervised(unsup_net, trainloader, device, epochs=20)

    # Load the last available weights based on the count
    # Path to the weights folder
    weights_dir = "../net_weights/unsup/"
    # Count the number of files that match the pattern
    file_count = len([f for f in os.listdir(weights_dir) if f.startswith("unsup_net_weights_") and f.endswith(".pth")])
    if file_count > 0:
        weight_path = f"{weights_dir}/unsup_net_weights_{file_count - 1}.pth"
        unsup_net.load_state_dict(torch.load(weight_path))
        print(f"Loaded weights from: {weight_path}")
    else:
        raise FileNotFoundError("No weight files found in the folder.")

    # Train the supervised model
    # # Initialize the supervised model using the encoder from the trained autoencoder
    sup_net = SupervisedNet(unsup_net)
    sup_net.to(device)
    train_supervised(sup_net, trainloader, device, epochs=20)
