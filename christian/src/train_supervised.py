import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_gabor_data  # Importing the data loading function from dataset.py
from Net import Net, SupervisedNet  # Import the autoencoder and supervised model from Net.py
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import numpy as np

def train_supervised(model, trainloader, device, epochs=15):
    # Define the loss function specific for supervised learning
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for classification
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    model.train()

    # Create the folder for saving results if it doesn't exist
    results_dir = "../epochs_results"
    os.makedirs(results_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist

    # Initialize the plots for real-time visualization
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots: 1 for Loss, 1 for Accuracy
    ax1.set_title("Supervised Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_title("Supervised Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")

    # Create plot lines for loss and accuracy
    loss_values = []
    accuracy_values = []
    loss_line, = ax1.plot([], [], label="Loss", color="blue")
    accuracy_line, = ax2.plot([], [], label="Accuracy", color="green")
    ax1.legend()
    ax2.legend()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
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

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Compute average loss and accuracy for the epoch
        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        loss_values.append(avg_loss)
        accuracy_values.append(accuracy)

        print(f"Supervised epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Update the real-time plots
        clear_output(wait=True)  # Clear output for smooth updates

        # Update loss plot
        loss_line.set_xdata(range(1, len(loss_values) + 1))  # Update x values (epochs)
        loss_line.set_ydata(loss_values)  # Update y values (loss)
        ax1.relim()  # Recalculate axis limits
        ax1.autoscale_view()  # Autoscale the view to fit data

        # Update accuracy plot
        accuracy_line.set_xdata(range(1, len(accuracy_values) + 1))  # Update x values (epochs)
        accuracy_line.set_ydata(accuracy_values)  # Update y values (accuracy)
        ax2.relim()  # Recalculate axis limits
        ax2.autoscale_view()  # Autoscale the view to fit data

        plt.pause(0.1)  # Pause to display the updated plot

        # Save the trained model weights
        # Save the trained model weights
        save_dir = "../net_weights/unsup"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        weights_dir = "../net_weights/sup"
        os.makedirs(weights_dir, exist_ok=True)  # Automatically create the directory if it doesn't exist
        torch.save(model.state_dict(), "../net_weights/sup/sup_net_weights_" + str(epoch) + ".pth")
        print("sup_net model weights saved as sup_net_weights.pth'")

    # Keep the plots open after training
    plt.ioff()
    plt.close(fig)
    # Save the loss values as a NumPy array
    loss_file_path = os.path.join(results_dir, "sup_epoch_losses.npy")
    np.save(loss_file_path, np.array(loss_values))  # Save as .npy file
    print(f"Loss values saved as NumPy array at: {loss_file_path}")


    accuracy_file_path = os.path.join(results_dir, "sup_epoch_accuracy.npy")
    np.save(accuracy_file_path, np.array(accuracy_values))  # Save as .npy file
    print(f"Accuracy values saved as NumPy array at: {accuracy_file_path}")





if __name__ == "__main__":
    # Path to your Excel file
    # Define the relative path
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")

    # Load the data
    trainloader, valloader, testloader = load_gabor_data(excel_file, batch_size=64)

    # Initialize the net and load the lastest encoder weights
    unsup_net = Net()
    # Path to the weights folder
    weights_dir = "../net_weights/unsup/"
    # Count the number of files that match the pattern
    file_count = len([f for f in os.listdir(weights_dir) if f.startswith("unsup_net_weights_") and f.endswith(".pth")])

    # Load the last available weights based on the count
    if file_count > 0:
        weight_path = f"{weights_dir}/unsup_net_weights_{file_count - 1}.pth"
        unsup_net.load_state_dict(torch.load(weight_path))
        print(f"Loaded weights from: {weight_path}")
    else:
        raise FileNotFoundError("No weight files found in the folder.")


    # Initialize the supervised model using the encoder from the trained autoencoder
    sup_net = SupervisedNet(unsup_net)

    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sup_net.to(device)

    # Train the supervised model
    train_supervised(sup_net, trainloader, device, epochs=25)
