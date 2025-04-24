import os
import torch
import numpy as np
from dataset import load_gabor_data
from Net import Net, SupervisedNet


def cosine_distance_matrix(embeddings):
    """
    Compute the pairwise cosine distance matrix for a batch of embeddings.
    Returns: Upper triangular cosine distance matrix (1 - cosine similarity).
    """
    normalized = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)  # Normalize to unit vectors
    cosine_similarity = torch.matmul(normalized, normalized.T)  # Cosine similarity
    cosine_distance = 1 - cosine_similarity  # Convert similarity to distance
    return cosine_distance.triu(diagonal=1)  # Upper triangular matrix excluding diagonal


def process_activations(activations, labels):
    """
    Processes activations and labels to calculate the mean within-category and between-category
    distances.

    The function computes cosine distances within each category and average distances
    between different categories provided by the activations and associated labels.
    Both distance metrics are useful in analyzing clustering based on categories.

    :param activations: torch.Tensor
        A tensor containing activation outputs to perform the clustering analysis on.

    :param labels: torch.Tensor
        A tensor representing labels corresponding to each activation in the input tensor.

    :return: Tuple[float, float]
        A tuple containing:
        - The mean of within-category distances.
        - The mean of between-category distances.
    """
    activations = activations.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    within_distances = []
    between_distances = []

    category_dict = {}
    for i, label in enumerate(labels):
        if label not in category_dict:
            category_dict[label] = []
        category_dict[label].append(activations[i])

    for category, members in category_dict.items():
        members_tensor = torch.tensor(members)
        within_dist = cosine_distance_matrix(members_tensor).mean().item()  # Within-category distance
        within_distances.append(within_dist)

        for other_category, other_members in category_dict.items():
            if category != other_category:
                other_members_tensor = torch.tensor(other_members)
                between_dist = torch.cdist(members_tensor, other_members_tensor).mean().item()
                between_distances.append(between_dist)

    return np.mean(within_distances), np.mean(between_distances)


def evaluate_and_save_epochs(model, trainloader, device, weight_dir, num_epochs, save_prefix):
    """
    Evaluates a model across multiple epochs using provided training data, calculates
    within-category and between-category average distances of activations, and saves
    the results for each epoch as numpy files.

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
    :param save_prefix: Prefix for the filenames of the saved output within and
        between distances.
    :type save_prefix: str
    :return: None
    """
    # Results directory
    results_dir = "../epochs_results"
    os.makedirs(results_dir, exist_ok=True)

    # Lists to store distances for all epochs
    within_distances_all = []
    between_distances_all = []

    for epoch in range(num_epochs):
        print(f"Processing Epoch {epoch + 1}/{num_epochs}...")

        # Load model weights for the epoch
        if weight_dir.endswith("unsup"):
            weight_path = os.path.join(weight_dir, f"unsup_net_weights_{epoch}.pth")
        else:
            weight_path = os.path.join(weight_dir, f"sup_net_weights_{epoch}.pth")

        model.load_state_dict(torch.load(weight_path, map_location=device))

        activation_holder = []

        # Define the hook function to collect activations
        def hook_fn(module, input, output):
            activation_holder.append(output)

        # Register forward hook on the last encoder layer
        hook_handle = model.encoder[-1].register_forward_hook(hook_fn)

        within_distances = []
        between_distances = []

        model.eval()
        with torch.no_grad():
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                _ = model(images)  # Forward pass

                activations = activation_holder.pop()  # Get the activations
                within_avg, between_avg = process_activations(activations, labels)

                within_distances.append(within_avg)
                between_distances.append(between_avg)

        hook_handle.remove()  # Remove the hook after processing

        # Compute per-epoch averages
        epoch_within_avg = np.mean(within_distances)
        epoch_between_avg = np.mean(between_distances)

        print(f"Epoch {epoch + 1}: Within Avg: {epoch_within_avg:.4f}, Between Avg: {epoch_between_avg:.4f}")

        # Append to global lists
        within_distances_all.append(epoch_within_avg)
        between_distances_all.append(epoch_between_avg)


    # Save results as NumPy files
    within_file = os.path.join(results_dir, f"{save_prefix}_within_distances.npy")
    between_file = os.path.join(results_dir, f"{save_prefix}_between_distances.npy")
    np.save(within_file, np.array(within_distances_all))
    np.save(between_file, np.array(between_distances_all))

    print(f"Saved within-category distances to: {within_file}")
    print(f"Saved between-category distances to: {between_file}")


if __name__ == "__main__":
    # Paths to weight directories
    unsup_weight_dir = os.path.abspath("../net_weights/unsup")
    sup_weight_dir = os.path.abspath("../net_weights/sup")

    # Number of epochs
    num_unsup_epochs = 20
    num_sup_epochs = 20

    # Load data
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")
    trainloader, _, _ = load_gabor_data(excel_file, batch_size=64)


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate unsupervised model distances at each epoch
    print("Evaluating unsupervised model...")
    unsup_net = Net()
    unsup_net.to(device)
    evaluate_and_save_epochs(unsup_net, trainloader, device, unsup_weight_dir, num_unsup_epochs, "unsup")

    # Evaluate supervised model distances at each epoch
    print("Evaluating supervised model...")
    sup_net = SupervisedNet(unsup_net)  # Use unsup encoder weights
    sup_net.to(device)
    evaluate_and_save_epochs(sup_net, trainloader, device, sup_weight_dir, num_sup_epochs, "sup")
