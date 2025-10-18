import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_gabor_data  # Importing the data loading function from dataset.py
#from Net import Net, SupervisedNet  # Import the autoencoder and supervised model from Net.py
from Mlp_Net import Net, SupervisedNet  # Import the autoencoder and supervised model from Net.py

import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import numpy as np
import matplotlib
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring import load_shape_deform_data

#matplotlib.use("TkAgg")  # Replace with a backend that supports interactivity

def train_supervised(model, trainloader, device, epochs=15):
    """
    Trains a supervised model and saves model weights, batch-specific loss & accuracy.
    Real-time visualization of loss and accuracy is included.
    """
    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Loss for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    model.train()

    # Create necessary directories to save results
    results_dir = "../loss"
    weights_dir = "../net_weights/sup"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Store all results
    batch_loss_values = []
    batch_accuracy_values = []

    # Initialize the plots for real-time visualization
    plt.ion()  # Interactive plotting mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Two subplots: Loss, Accuracy

    # Loss plot configuration
    ax1.set_title("Supervised Training Loss")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss")
    loss_line, = ax1.plot([], [], label="Loss", color="blue")
    ax1.legend()

    # Accuracy plot configuration
    ax2.set_title("Supervised Training Accuracy")
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Accuracy (%)")
    accuracy_line, = ax2.plot([], [], label="Accuracy", color="green")
    ax2.legend()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}/{epochs}")
        running_loss, correct, total = 0.0, 0, 0

        # Store current epoch data
        epoch_losses = []
        epoch_accuracies = []

        for batch, (images, labels) in enumerate(trainloader, start=1):
            # Transfer data to the device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate individual batch metrics
            batch_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)

            # Update counters and lists
            running_loss += batch_loss
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            batch_loss_values.append(batch_loss)
            batch_accuracy_values.append(batch_accuracy)
            epoch_losses.append(batch_loss)
            epoch_accuracies.append(batch_accuracy)

            # Save model weights for the current batch
            batch_weight_path = os.path.join(
                weights_dir, f"sup_net_weights_epoch_{epoch}_batch_{batch}.pth"
            )
            torch.save(model.state_dict(), batch_weight_path)

            # Save individual batch losses and accuracies to files
            batch_loss_file = os.path.join(
                results_dir, f"sup_loss_epoch_{epoch}_batch_{batch}.npy"
            )
            batch_accuracy_file = os.path.join(
                results_dir, f"sup_accuracy_epoch_{epoch}_batch_{batch}.npy"
            )
            np.save(batch_loss_file, np.array(batch_loss))
            np.save(batch_accuracy_file, np.array(batch_accuracy))

            print(f"Saved batch loss to: {batch_loss_file}")
            print(f"Saved batch accuracy to: {batch_accuracy_file}")

            # Update real-time plots
            clear_output(wait=True)  # Clear console output for cleaner updates
            loss_line.set_xdata(range(1, len(batch_loss_values) + 1))
            loss_line.set_ydata(batch_loss_values)
            ax1.relim()
            ax1.autoscale_view()

            accuracy_line.set_xdata(range(1, len(batch_accuracy_values) + 1))
            accuracy_line.set_ydata(batch_accuracy_values)
            ax2.relim()
            ax2.autoscale_view()

            plt.pause(0.1)  # Brief pause for visualization update

            # Log batch metrics
            print(
                f"Epoch [{epoch}/{epochs}], Batch [{batch}/{len(trainloader)}], "
                f"Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.2f}%"
            )

        # Calculate and log the per-epoch averages
        avg_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch}/{epochs}] completed with Avg Loss: {avg_loss:.4f}, "
            f"Avg Accuracy: {epoch_accuracy:.2f}%"
        )

        # Save per-epoch losses and accuracies
        epoch_loss_file = os.path.join(results_dir, f"sup_losses_epoch_{epoch}.npy")
        epoch_accuracy_file = os.path.join(results_dir, f"sup_accuracies_epoch_{epoch}.npy")
        np.save(epoch_loss_file, np.array(epoch_losses))
        np.save(epoch_accuracy_file, np.array(epoch_accuracies))
        print(f"Saved epoch loss to: {epoch_loss_file}")
        print(f"Saved epoch accuracy to: {epoch_accuracy_file}")

    # Save all batch-level results to master files
    master_loss_file = os.path.join(results_dir, "sup_all_batch_losses.npy")
    master_accuracy_file = os.path.join(results_dir, "sup_all_batch_accuracies.npy")
    np.save(master_loss_file, np.array(batch_loss_values))
    np.save(master_accuracy_file, np.array(batch_accuracy_values))
    print(f"All batch loss values saved at: {master_loss_file}")
    print(f"All batch accuracy values saved at: {master_accuracy_file}")

    # Finalize the plots
    # Finalize the plot
    plt.ioff()  # Disable interactive mode
    plt.close(fig)  # Explicitly close the specific figure




if __name__ == "__main__":
    # Path to your Excel file
    # Define the relative path
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")

    # Load the data
    #trainloader, valloader, testloader = load_gabor_data(excel_file, batch_size=64)
    #trainloader, valloader, testloader, full_dataset = load_many_arcs_data(m_arcs_per_class=5,gap_frac=0.1)
    #trainloader, valloader, testloader, full_dataset = load_image_ring_data(m_arcs_per_class=16,gap_frac=0.3)
    # ---- Configurable parameters ----
    m_arcs_per_class = 2
    gap_frac = (m_arcs_per_class * 0.2) / np.pi
    mode = "indep_phase"  # "phase" | "mode" | "indep_phase"

    # Harmonics and amplitudes
    k1, k2, k3, k4 = 3, 3, None, None
    a1, a2, a3, a4 = 0.1, 0.1, 0.0, 0.0
    alpha_x, beta_y, gamma_x, delta_y = 0.1, 0.1, 0.1, 0.1

    # Rendering parameters
    image_size = 128
    ring_radius_px = 40
    nA = 200
    nB = nA

    # ---- Create dataset ----
    trainloader, valloader, testloader, ds = load_shape_deform_data(nA=nA, nB=nB,
                                                                    m_arcs_per_class=m_arcs_per_class,
                                                                    gap_frac=gap_frac,
                                                                    mode=mode,
                                                                    k1=k1, k2=k2, k3=k3, k4=k4,
                                                                    a1=a1, a2=a2, a3=a3, a4=a4,
                                                                    alpha_x=alpha_x, beta_y=beta_y,
                                                                    gamma_x=gamma_x, delta_y=delta_y,
                                                                    image_size=image_size,
                                                                    ring_radius_px=ring_radius_px,
                                                                    )
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