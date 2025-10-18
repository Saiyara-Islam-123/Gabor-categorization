import os
import torch
import numpy as np
from dataset import load_gabor_data
#from Net import Net, SupervisedNet
from Mlp_Net import Net, SupervisedNet
from rings_no_overlap import load_many_arcs_data





def evaluate_and_save_epochs(model, trainloader, device, weight_dir, num_epochs, save_prefix):
    """
    Evaluates a model across multiple epochs using provided training data and saves
    the layer activations for each epoch.

    :param model: The deep learning model to evaluate.
    :type model: torch.nn.Module
    :param trainloader: DataLoader providing training data.
    :type trainloader: torch.utils.data.DataLoader
    :param device: The device to run the model on (e.g., 'cuda' or 'cpu').
    :type device: torch.device
    :param weight_dir: Directory path where the model's weights for different epochs
        are stored.
    :type weight_dir: str
    :param num_epochs: The number of epochs to evaluate the model for.
    :type num_epochs: int
    :param save_prefix: Prefix for the filenames of the saved activations.
    :type save_prefix: str
    :return: None
    """
    # Results directory
    results_dir = "../epochs_results"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Processing Epoch {epoch + 1}/{num_epochs}...")

        # Load model weights for the epoch
        if weight_dir.endswith("unsup"):
            weight_path = os.path.join(weight_dir, f"unsup_net_weights_{epoch}.pth")
        else:
            weight_path = os.path.join(weight_dir, f"sup_net_weights_{epoch}.pth")

        model.load_state_dict(torch.load(weight_path, map_location=device))

        all_activations = []
        all_labels = []

        # Define the hook function to collect activations
        def hook_fn(module, input, output):
            all_activations.append(output.cpu().detach())

        # Register forward hook on the last encoder layer
        hook_handle = model.encoder[-1].register_forward_hook(hook_fn)

        model.eval()
        with torch.no_grad():
            for images, labels in trainloader:
                images = images.to(device)
                _ = model(images)  # Forward pass
                all_labels.append(labels.cpu())

        hook_handle.remove()  # Remove the hook after processing

        # Concatenate all batches
        epoch_activations = torch.cat(all_activations, dim=0)
        epoch_labels = torch.cat(all_labels, dim=0)

        # Save activations and labels
        activation_file = os.path.join(results_dir, f"{save_prefix}_activations_epoch_{epoch}.npy")
        labels_file = os.path.join(results_dir, f"{save_prefix}_labels_epoch_{epoch}.npy")
        np.save(activation_file, epoch_activations.numpy())
        np.save(labels_file, epoch_labels.numpy())

        print(f"Saved activations for epoch {epoch + 1}")


if __name__ == "__main__":
    # Paths to weight directories
    unsup_weight_dir = os.path.abspath("../net_weights/unsup")
    sup_weight_dir = os.path.abspath("../net_weights/sup")

    # Number of epochs
    num_unsup_epochs = 20
    num_sup_epochs = 20

    # Load data
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation_with_control.xlsx")
    #trainloader, _, _ = load_gabor_data(excel_file, batch_size=64)
    trainloader, valloader, testloader, full_dataset = load_many_arcs_data(m_arcs_per_class=5,gap_frac=0.3)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save activations for unsupervised model
    print("Processing unsupervised model activations...")
    unsup_net = Net()
    unsup_net.to(device)
    evaluate_and_save_epochs(unsup_net, trainloader, device, unsup_weight_dir, num_unsup_epochs, "unsup")

    # Save activations for supervised model
    print("Processing supervised model activations...")
    sup_net = SupervisedNet(unsup_net)  # Use unsup encoder weights
    sup_net.to(device)
    evaluate_and_save_epochs(sup_net, trainloader, device, sup_weight_dir, num_sup_epochs, "sup")
