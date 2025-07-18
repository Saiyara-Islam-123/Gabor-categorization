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

def train_supervised(model, trainloader, device, lr, epochs=15):
    """
    Trains a given model using supervised learning with a provided dataloader, device,
    and a specified number of epochs. The function uses the CrossEntropyLoss for
    classification tasks, and the Adam optimizer for parameter updates. During training,
    it visualizes the loss and accuracy over epochs using real-time plots and saves the
    model's weights after each epoch. Additionally, the computed loss and accuracy values
    are saved for post-training analysis.

    :param model: Neural network model to be trained
    :type model: torch.nn.Module
    :param trainloader: DataLoader providing batches of training data
    :type trainloader: torch.utils.data.DataLoader
    :param device: Device on which the model and data will be loaded (e.g., 'cpu' or 'cuda')
    :type device: torch.device
    :param epochs: Number of training epochs; defaults to 15
    :type epochs: int, optional
    :return: None
    :rtype: None
    """
    # Define the loss function specific for supervised learning
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model.train()

    # Create the folder for saving results if it doesn't exist
    results_dir = "../epochs_results"
    os.makedirs(results_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist

    # Initialize the plots for real-time visualization
    #plt.ion()
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots: 1 for Loss, 1 for Accuracy
    #ax1.set_title("Supervised Training Loss")
    #ax1.set_xlabel("Epoch")
    #ax1.set_ylabel("Loss")
    #ax2.set_title("Supervised Training Accuracy")
    #ax2.set_xlabel("Epoch")
    #ax2.set_ylabel("Accuracy (%)")

    # Create plot lines for loss and accuracy
    loss_values = []
    accuracy_values = []
    #loss_line, = ax1.plot([], [], label="Loss", color="blue")
    #accuracy_line, = ax2.plot([], [], label="Accuracy", color="green")
    #ax1.legend()
    #ax2.legend()

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
            zero, zero_one, one = sampled_all_distance(encoder_outputs, labels)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            accuracy_values.append(accuracy)
            print(accuracy)

            weights_dir = "../net_weights/sup"
            os.makedirs(weights_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist
            torch.save(model.state_dict(), f"../net_weights/sup/sup_net_weights_ lr={lr} "+str(epoch)+  " " + str(batch) +".pth")
            print("sup_net model weights saved as sup_net_weights.pth'")
            batch += 1

        # Compute average loss and accuracy for the epoch
        avg_loss = running_loss / len(trainloader)
        loss_values.append(avg_loss)

        print(f"Supervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f},")

        # Update the real-time plots
        #clear_output(wait=True)  # Clear output for smooth updates

        # Update loss plot
        #loss_line.set_xdata(range(1, len(loss_values) + 1))  # Update x values (epochs)
        #loss_line.set_ydata(loss_values)  # Update y values (loss)
        #ax1.relim()  # Recalculate axis limits
        #ax1.autoscale_view()  # Autoscale the view to fit data

        # Update accuracy plot
        #accuracy_line.set_xdata(range(1, len(accuracy_values) + 1))  # Update x values (epochs)
        #accuracy_line.set_ydata(accuracy_values)  # Update y values (accuracy)
        #ax2.relim()  # Recalculate axis limits
        #ax2.autoscale_view()  # Autoscale the view to fit data

        #plt.pause(0.1)  # Pause to display the updated plot

        # Save the trained model weights
        # Save the trained model weights


    # Keep the plots open after training
    #plt.ioff()
    #plt.close(fig)
    # Save the loss values as a NumPy array
    loss_file_path = os.path.join(results_dir, "sup_epoch_losses.npy")
    np.save(loss_file_path, np.array(loss_values))  # Save as .npy file
    print(f"Loss values saved as NumPy array at: {loss_file_path}")

    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]
    df["acc"] = accuracy_values
    df.to_csv(f"LR={lr}, Distance every batch sup, 2 epochs.csv", index=False)


    accuracy_file_path = os.path.join(results_dir, "sup_epoch_accuracy.npy")
    np.save(accuracy_file_path, np.array(accuracy_values))  # Save as .npy file
    print(f"Accuracy values saved as NumPy array at: {accuracy_file_path}")

def train_supervised_control(model, main_trainloader, device, lr, epochs, side_train_loader, title, weights_dir, is_control):
    """
    Trains a given model using supervised learning with a provided dataloader, device,
    and a specified number of epochs. The function uses the CrossEntropyLoss for
    classification tasks, and the Adam optimizer for parameter updates. During training,
    it visualizes the loss and accuracy over epochs using real-time plots and saves the
    model's weights after each epoch. Additionally, the computed loss and accuracy values
    are saved for post-training analysis.

    :param model: Neural network model to be trained
    :type model: torch.nn.Module
    :param trainloader: DataLoader providing batches of training data
    :type trainloader: torch.utils.data.DataLoader
    :param device: Device on which the model and data will be loaded (e.g., 'cpu' or 'cuda')
    :type device: torch.device
    :param epochs: Number of training epochs; defaults to 15
    :type epochs: int, optional
    :return: None
    :rtype: None
    """
    # Define the loss function specific for supervised learning
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model.train()

    # Create the folder for saving results if it doesn't exist
    results_dir = "../epoch_results_control"
    os.makedirs(results_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist


    loss_values = []
    accuracy_values_main = []
    accuracy_values_side = []

    avg_distances = {}
    avg_distances[(0, 0)] = []
    avg_distances[(0, 1)] = []
    avg_distances[(1, 1)] = []


    for epoch in range(epochs):
        running_loss = 0.0
        correct_main = 0
        correct_side = 0
        total_main = 0
        total_side = 0
        batch = 0

        for images_main, labels_main in main_trainloader:
            # Prepare the images and labels
            print(epoch, batch)
            images_main = images_main.to(device)  # Move input images to the same device as the model

            labels_main = labels_main.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images_main)
            loss_true = criterion(outputs, labels_main)

            # Backward pass and optimize
            loss_true.backward()
            optimizer.step()

            running_loss += loss_true.item()
            print("Loss:", loss_true.item())

            _, predicted_main = torch.max(outputs.data, 1)
            total_main += labels_main.size(0)
            correct_main += (predicted_main == labels_main).sum().item()

            accuracy_main = 100 * correct_main / total_main
            accuracy_values_main.append(accuracy_main)
            print("Main: ", accuracy_main)

            encoder_outputs_main = model.encoder_output
            model.remove_encoder_output()

            ###################################################################

            images_side, labels_side = next(itertools.cycle(side_train_loader))
            outputs_side = model(images_side)

            _, predicted_side = torch.max(outputs_side.data, 1)
            total_side += labels_side.size(0)
            correct_side += (predicted_side == labels_side).sum().item()

            accuracy_true = 100 * correct_side / total_side
            accuracy_values_side.append(accuracy_true)
            print("Side: ",accuracy_true)

            encoder_outputs_side = model.encoder_output
            model.remove_encoder_output()

            ####################################################################
            if not is_control:
                zero, zero_one, one = sampled_all_distance(encoder_outputs_main, labels_main)
            else:
                zero, zero_one, one = sampled_all_distance(encoder_outputs_side, labels_side)

            avg_distances[(0, 0)].append(zero)
            avg_distances[(0, 1)].append(zero_one)
            avg_distances[(1, 1)].append(one)

            torch.save(model.state_dict(), f"../net_weights/{weights_dir}/sup_net_weights_lr={lr} "+str(epoch)+  " " + str(batch) +".pth")
            print("sup_net model weights saved as sup_net_weights.pth'")
            batch += 1


    loss_file_path = os.path.join(results_dir, "sup_epoch_losses.npy")
    np.save(loss_file_path, np.array(loss_values))  # Save as .npy file
    print(f"Loss values saved as NumPy array at: {loss_file_path}")

    df = pd.DataFrame()
    df["within 0"] = avg_distances[(0, 0)]
    df["within 1"] = avg_distances[(1, 1)]
    df["between"] = avg_distances[(0, 1)]

    if not is_control:
        df["acc size"] = accuracy_values_side
        df["acc freq"] = accuracy_values_main

    else:
        df["acc freq"] = accuracy_values_side
        df["acc size"] = accuracy_values_main

    df.to_csv(f"LR={lr} {title} Distance every batch sup.csv", index=False)



if __name__ == "__main__":
    print()