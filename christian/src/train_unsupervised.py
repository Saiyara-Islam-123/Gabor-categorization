import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_gabor_data  # Importing the data loading function from dataset.py
from Net import Net
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import numpy as np
from sampling import *
import pandas as pd

def train_unsupervised(model, trainloader, device, epochs=5):
    """
    Trains an unsupervised model (e.g., autoencoder) using a specified dataset and parameters.
    This function uses Mean Squared Error (MSE) loss for reconstruction and updates the model's
    parameters using the Adam optimizer. The training progress, including the real-time loss plot,
    is updated during each epoch. Additionally, the model's weights and epoch loss values are
    periodically saved to specified directories.

    :param model: The PyTorch model to be trained.
    :type model: torch.nn.Module
    :param trainloader: DataLoader providing the training data, which should return batches of images.
    :type trainloader: torch.utils.data.DataLoader
    :param device: The device on which computations will be performed (e.g., 'cuda' or 'cpu').
    :type device: str
    :param epochs: The number of training epochs. Default is 5.
    :type epochs: int, optional
    :return: None
    """
    # Define the loss function specific for autoencoder
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    loss_values = []

    # Create the folder for saving results if it doesn't exist
    results_dir = "../epochs_results"
    os.makedirs(results_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist

    # Initialize the plot for real-time visualization
    #plt.ion()
    #fig, ax = plt.subplots()
    #ax.set_title("Training Loss Over Epochs")
    #ax.set_xlabel("Epoch")
    #ax.set_ylabel("Loss")
    #loss_line, = ax.plot([], [], label="Loss", color="blue")  # Create the line for loss
    #ax.legend()  # Add legend once
    avg_distances = {}
    avg_distances[(0,0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images = images.to(device)  # Move input images to the same device as the model

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)


            loss = criterion(outputs, images)

            avg_distances[(0,0)].append( sampled_avg_distance(pair=(0, 0), X=outputs, y=labels))
            avg_distances[(0, 1)].append(sampled_avg_distance(pair=(0, 1), X=outputs, y=labels))
            avg_distances[(1, 1)].append(sampled_avg_distance(pair=(1, 1), X=outputs, y=labels))


            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        loss_values.append(avg_loss)
        print(f"Unsupervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # Update the real-time plot
        #clear_output(wait=True)  # Clear the previous output
        #loss_line.set_xdata(range(1, len(loss_values) + 1))  # Update x values (epochs)
        #loss_line.set_ydata(loss_values)  # Update y values (loss)

        #ax.relim()  # Recalculate limits
        #ax.autoscale_view()  # Autoscale to fit new data

        #plt.pause(0.1)  # Pause to display the updated plot

        weights_dir = "../net_weights/unsup"
        os.makedirs(weights_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist
        torch.save(model.state_dict(), "../net_weights/unsup/unsup_net_weights_"+str(epoch)+".pth")
        print("unsup_net model weights saved as 'unsup_net_weights.pth'")

    # Keep the plot open after training
    #plt.ioff()
    #plt.close(fig)
    # Save the loss values as a NumPy array
    loss_file_path = os.path.join(results_dir, "unsup_epoch_losses.npy")
    np.save(loss_file_path, np.array(loss_values))  # Save as .npy file
    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0,0)]
    df["within 1"] = avg_distances[(1,1)]
    df["between"] = avg_distances[(0,1)]
    df.to_csv("Distance per batch unsup.csv", index=False)

    print(f"Loss values saved as NumPy array at: {loss_file_path}")


if __name__ == "__main__":
    # Path to your Excel file
    # Define the relative path
    excel_file = "categorisation.xlsx"

    # Load the data
    trainloader, valloader, testloader = load_gabor_data(excel_file,batch_size=64)

    # Initialize the autoencoder model
    unsup_net = Net()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the model
    train_unsupervised(unsup_net, trainloader, device, epochs=15)
