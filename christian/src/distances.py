import os
import torch
import numpy as np
from dataset import load_gabor_data
from Net import Net, SupervisedNet
#from Mlp_Net import Net, SupervisedNet
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring_2 import load_shape_deform_data

def cosine_distance_matrix(embeddings1, embeddings2=None):
    """
    Compute the pairwise cosine distance matrix.
    - If `embeddings2` is None, computes distances within `embeddings1`.
    - If `embeddings2` is provided, computes distances between `embeddings1` and `embeddings2`.

    Args:
        embeddings1 (torch.Tensor): The first set of embeddings.
        embeddings2 (torch.Tensor, optional): The second set of embeddings. Defaults to None.

    Returns:
        torch.Tensor: The cosine distance matrix.
    """
    normalized1 = embeddings1 / torch.norm(embeddings1, p=2, dim=1, keepdim=True)  # Normalize to unit vectors
    if embeddings2 is None:
        # Case: Within-tensor distances
        cosine_similarity = torch.matmul(normalized1, normalized1.T)  # Cosine similarity
    else:
        # Case: Between-tensor distances
        normalized2 = embeddings2 / torch.norm(embeddings2, p=2, dim=1, keepdim=True)
        cosine_similarity = torch.matmul(normalized1, normalized2.T)  # Cosine similarity

    cosine_distance = 1 - cosine_similarity  # Convert similarity to distance
    return cosine_distance
  
import torch

import torch
import torch.nn.functional as F

def euclidean_distance_matrix(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor = None,
    normalize: bool = True,
    eps: float = 1e-8
) :
    """
    Compute the pairwise Euclidean distance matrix with optional L2 normalization.

    Args:
        embeddings1 (torch.Tensor): First set of embeddings of shape [N, D].
        embeddings2 (torch.Tensor, optional): Second set of embeddings of shape [M, D].
        normalize (bool): If True, each vector is L2-normalized before distance calc.
        eps (float): Small constant to avoid division by zero during normalization.

    Returns:
        torch.Tensor: Matrix of Euclidean distances.
            Shape [N, N] if embeddings2 is None, else [N, M].
    """
    if normalize:
        embeddings1 = F.normalize(embeddings1, p=2, dim=1, eps=eps)
        if embeddings2 is not None:
            embeddings2 = F.normalize(embeddings2, p=2, dim=1, eps=eps)

    if embeddings2 is None:
        # Within-set distances
        sq_norms = (embeddings1 ** 2).sum(dim=1, keepdim=True)           # [N, 1]
        dist_sq = sq_norms + sq_norms.T - 2 * embeddings1 @ embeddings1.T
    else:
        # Between-set distances
        sq_norms1 = (embeddings1 ** 2).sum(dim=1, keepdim=True)          # [N, 1]
        sq_norms2 = (embeddings2 ** 2).sum(dim=1, keepdim=True)          # [M, 1]
        dist_sq = sq_norms1 + sq_norms2.T - 2 * embeddings1 @ embeddings2.T

    dist_sq = torch.clamp(dist_sq, min=0.0)  # avoid negative values from round-off
    return dist_sq


def compute_distances(activations, labels):
    """
    Compute the average within-class and between-class Euclidean distances.

    Args
    ----
    activations : torch.Tensor
        Latent vectors for a batch of samples, shape [B, D].
    labels : torch.Tensor
        Integer class labels for each sample, shape [B].

    Returns
    -------
    tuple of four floats:
        (mean_within_class0, mean_within_class1,
         mean_within_overall, mean_between_overall)
    """

    # Move data to CPU and drop the computational graph
    activations = activations.detach().cpu()
    labels      = labels.detach().cpu()

    # Lists to accumulate distances for different aggregates
    within_distances       = []  # all classes pooled
    between_distances      = []  # all between–class pairs
    within_distances_cat0  = []  # only class 0
    within_distances_cat1  = []  # only class 1

    # ------------------------------------------------------------------
    # Group the embeddings by their label so we can compare
    # within each class and between classes.
    # category_dict[label] -> list of activation vectors for that label
    # ------------------------------------------------------------------
    category_dict = {}
    for i, lab in enumerate(labels.tolist()):
        category_dict.setdefault(lab, []).append(activations[i])

    # ------------------------------------------------------------------
    # Loop over each class to compute:
    #   • mean pairwise distance among its own members
    #   • mean distance to every other class
    # ------------------------------------------------------------------
    for category, members in category_dict.items():
        members_tensor = torch.stack(members)            # shape [N, D]

        # Pairwise squared distances inside this class
        D2 = euclidean_distance_matrix(members_tensor)   # [N, N]

        # ---- Correct mean over unique pairs ----
        # D2 is symmetric with zeros on the diagonal; we only want the
        # upper-triangle (each unordered pair counted once).
        N  = D2.size(0)
        iu = torch.triu_indices(N, N, offset=1)          # indices above diagonal
        within_vals = D2[iu[0], iu[1]]                   # distances for unique pairs
        within_dist = within_vals.mean().item()

        # Store class-specific within-class means
        if category == 0:
            within_distances_cat0.append(within_dist)
        else:
            within_distances_cat1.append(within_dist)

        # Store for overall within-class mean (all classes pooled)
        within_distances.append(within_dist)

        # ---- Between-class distances ----
        # For each other class, compute the mean distance
        # between this class and that other class.
        for other_category, other_members in category_dict.items():
            if category != other_category:
                other_tensor = torch.stack(other_members)
                # Full matrix of distances between the two sets, take global mean
                between_dist = euclidean_distance_matrix(
                    members_tensor, other_tensor
                ).mean().item()
                between_distances.append(between_dist)

    # Aggregate all the collected statistics into scalars
    return (np.mean(within_distances_cat0),   # mean within-class distance for label 0
            np.mean(within_distances_cat1),   # mean within-class distance for label 1
            np.mean(within_distances),        # mean within-class distance across all labels
            np.mean(between_distances))       # mean distance between different classes

def evaluate_and_save_epochs_and_batches(model, trainloader, device, weight_dir, num_epochs, num_batches, save_prefix):
    """
    Evaluates a model across multiple epochs and batches using training data, calculates
    within-category and between-category average distances, and saves results for every
    epoch and batch.

    :param model: The deep learning model to evaluate.
    :type model: torch.nn.Module
    :param trainloader: DataLoader providing training data.
    :type trainloader: torch.utils.data.DataLoader
    :param device: The device to run the model on (e.g., 'cuda' or 'cpu').
    :type device: torch.device
    :param weight_dir: Directory path where the model's weights for different epochs and batches are stored.
    :type weight_dir: str
    :param num_epochs: The number of epochs to evaluate the model for.
    :type num_epochs: int
    :param num_batches: The number of batches to evaluate per epoch.
    :type num_batches: int
    :param save_prefix: Prefix for the filenames of the saved results.
    :type save_prefix: str
    :return: None
    """
    # Results directory
    results_dir = "../distances"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Processing Epoch {epoch + 1}/{num_epochs}...")

        within_distances_cat0_epoch = []
        within_distances_cat1_epoch = []
        within_distances_epoch = []
        between_distances_epoch = []

        for batch in range(1, num_batches + 1):
            print(f"  Processing Batch {batch}/{num_batches}...")

            # Load model weights for the epoch and batch
            if weight_dir.endswith("unsup"):
                weight_path = os.path.join(weight_dir, f"unsup_net_weights_epoch_{epoch}_batch_{batch}.pth")
            else:
                weight_path = os.path.join(weight_dir, f"sup_net_weights_epoch_{epoch}_batch_{batch}.pth")

            if not os.path.exists(weight_path):
                print(f"    Weight file {weight_path} not found. Skipping batch.")
                continue

            # Load weights into the model
            model.load_state_dict(torch.load(weight_path, map_location=device))

            activation_holder = []
            all_within_cat0 = []
            all_within_cat1 = []
            all_within = []
            all_between = []

            # Define the hook function to collect activations
            def hook_fn(module, input, output):
                activation_holder.append(output)

            # Register forward hook on the last encoder layer
            hook_handle = model.encoder[-1].register_forward_hook(hook_fn)

            model.eval()
            with torch.no_grad():
                for images, labels in trainloader:
                    images, labels = images.to(device), labels.to(device)
                    _ = model(images)  # Forward pass
                    activations = activation_holder.pop()  # Get activations for one pass

                    # Compute distances for this batch
                    within_cat0, within_cat1, within_avg, between_avg = compute_distances(activations, labels)

                    all_within_cat0.append(within_cat0)
                    all_within_cat1.append(within_cat1)
                    all_within.append(within_avg)
                    all_between.append(between_avg)

                # Compute batch-level averages
                batch_within_cat0 = np.mean(all_within_cat0)
                batch_within_cat1 = np.mean(all_within_cat1)
                batch_within_avg = np.mean(all_within)
                batch_between_avg = np.mean(all_between)

                print(f"    Batch {batch}: Within Avg: {batch_within_avg:.4f}, Between Avg: {batch_between_avg:.4f}")

                # Save batch-level results
                np.save(
                    os.path.join(results_dir, f"{save_prefix}_within_distances_cat0_epoch_{epoch}_batch_{batch}.npy"),
                    np.array(batch_within_cat0)
                )
                np.save(
                    os.path.join(results_dir, f"{save_prefix}_within_distances_cat1_epoch_{epoch}_batch_{batch}.npy"),
                    np.array(batch_within_cat1)
                )
                np.save(
                    os.path.join(results_dir, f"{save_prefix}_within_distances_epoch_{epoch}_batch_{batch}.npy"),
                    np.array(batch_within_avg)
                )
                np.save(
                    os.path.join(results_dir, f"{save_prefix}_between_distances_epoch_{epoch}_batch_{batch}.npy"),
                    np.array(batch_between_avg)
                )

                # Accumulate for epoch-level results
                within_distances_cat0_epoch.append(batch_within_cat0)
                within_distances_cat1_epoch.append(batch_within_cat1)
                within_distances_epoch.append(batch_within_avg)
                between_distances_epoch.append(batch_between_avg)

            hook_handle.remove()  # Remove the hook after processing the batch

        # Compute epoch-level averages
        epoch_within_cat0_avg = np.mean(within_distances_cat0_epoch)
        epoch_within_cat1_avg = np.mean(within_distances_cat1_epoch)
        epoch_within_avg = np.mean(within_distances_epoch)
        epoch_between_avg = np.mean(between_distances_epoch)

        print(f"Epoch {epoch + 1}: Within Avg: {epoch_within_avg:.4f}, Between Avg: {epoch_between_avg:.4f}")

        # Save epoch-level results
        np.save(
            os.path.join(results_dir, f"{save_prefix}_within_distances_cat0_epoch_{epoch}.npy"),
            np.array(within_distances_cat0_epoch)
        )
        np.save(
            os.path.join(results_dir, f"{save_prefix}_within_distances_cat1_epoch_{epoch}.npy"),
            np.array(within_distances_cat1_epoch)
        )
        np.save(
            os.path.join(results_dir, f"{save_prefix}_within_distances_epoch_{epoch}.npy"),
            np.array(within_distances_epoch)
        )
        np.save(
            os.path.join(results_dir, f"{save_prefix}_between_distances_epoch_{epoch}.npy"),
            np.array(between_distances_epoch)
        )


if __name__ == "__main__":
    # Paths to weight directories
    unsup_weight_dir = os.path.abspath("../net_weights/unsup")
    sup_weight_dir = os.path.abspath("../net_weights/sup")

    # Number of epochs and batches
    num_unsup_epochs =20
    num_sup_epochs = 10
    num_batches = 7

    # Load data
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles", "categorisation_with_control.xlsx")
    #trainloader, _, _ = load_gabor_data(excel_file, batch_size=64)
    #trainloader, valloader, testloader, full_dataset = load_many_arcs_data(m_arcs_per_class=5,gap_frac=0.3)
    #trainloader, valloader, testloader, full_dataset = load_image_ring_data(m_arcs_per_class=16,gap_frac=0.3)
    # ---- Configurable parameters ----
    # ---- Configurable parameters ----
    nA = 1000
    nB = nA
    m_arcs_per_class = 2
    gap_frac = 0.5
    phase_deg = 0.0

    # Amplitude ring params (independent)
    amp_m_arcs_per_class = 6
    amp_gap_frac = 0.15

    # Amplitude ring controls
    amp_ring_radius = 1  # smaller = subtler deformation
    amp_scale_a1 = 0.2
    amp_scale_a2 = 0.3

    difficulty_sharp = 0.2  # try 0.3–0.8; 0 = no sharpening difference

    image_size = 128
    ring_radius_px = 40

    k1, k2 = 3, 5
    a1, a2 = 0.1, 0.2

    # Phase config
    phase_mode = "independent"  # "independent" | "fixed"
    phase_fixed_values = (0, 0)  # used iff phase_mode == "fixed"

    # Amplitudes config
    amp_mode = "ring_shared_arcs"  # "fixed" | "independent_uniform" | "ring_shared_arcs"
    amp_class_coupling = "parity"  # "none" | "parity"
    # ---- Create dataset ----
    trainloader, valloader, testloader, ds = load_shape_deform_data(
        nA=nA, nB=nB,
        m_arcs_per_class=m_arcs_per_class, gap_frac=gap_frac, phase_deg=phase_deg,
        image_size=image_size, ring_radius_px=ring_radius_px,
        k1=k1, k2=k2,
        a1=a1, a2=a2,
        mode="indep_phase",
        alpha_x=0.1, beta_y=0.1,
        intensity=1.0, bg=0.0, norm="max", global_rot="none",
        amp_mode=amp_mode, amp_class_coupling=amp_class_coupling,
        difficulty_sharp=difficulty_sharp,
        amp_ring_radius=amp_ring_radius,
        amp_scale_a1=amp_scale_a1, amp_scale_a2=amp_scale_a2,
        amp_m_arcs_per_class=amp_m_arcs_per_class, amp_gap_frac=amp_gap_frac,
        phase_mode=phase_mode, phase_fixed_values=phase_fixed_values,
        batch_size=256, seed=42
    )
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate and save distances for unsupervised model
    print("Processing unsupervised model...")
    unsup_net = Net()
    unsup_net.to(device)
    evaluate_and_save_epochs_and_batches(unsup_net, trainloader, device, unsup_weight_dir, num_unsup_epochs, num_batches, "unsup")

    # Evaluate and save distances for supervised model
    print("Processing supervised model...")
    sup_net = SupervisedNet(unsup_net)
    sup_net.to(device)
    evaluate_and_save_epochs_and_batches(sup_net, trainloader, device, sup_weight_dir, num_sup_epochs, num_batches, "sup")
