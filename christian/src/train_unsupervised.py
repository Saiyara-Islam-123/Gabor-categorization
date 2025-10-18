import torch
import torch.nn as nn
import torch.optim as optim
#from dataset import load_gabor_data  #Importing the data loading function from dataset.py
from Net import Net
#from Mlp_Net import Net
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os
import numpy as np
import matplotlib
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring import load_shape_deform_data

import torch
import torch.nn.functional as F

matplotlib.use("TkAgg")  # Replace with a backend that supports interactivity

# Helper functions to add a constraint on the loss,
# ensuring that within distance = between distance on the

def _mean_pairwise_sqdist(z):
    # z: tensor of shape [B, D], where B = batch size, D = latent dim.
    # Goal: compute the full matrix of pairwise squared Euclidean distances
    # between all rows in z, using the identity:
    #   ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    G = z @ z.t()                         # [B, B] Gram matrix of inner products a·b
    q = (z * z).sum(dim=1, keepdim=True)  # [B, 1] vector of squared norms ||a||^2
    D2 = q + q.t() - 2 * G                # [B, B] pairwise squared distances
    # Diagonal is exactly zero (distance of each point to itself).
    # We won't use it when computing "within" distances (we mask it out).
    return D2

def latent_within_between(z, y, normalize=True, eps=1e-8):
    """
    Returns mean *squared* distances (within, between) using your unified function.
    """
    # Use the same metric as everywhere else:
    D2 = euclidean_distance_matrix(z, normalize=normalize)   # [B,B], squared if your fn returns dist_sq

    B = z.size(0)
    same = y.unsqueeze(0).eq(y.unsqueeze(1))
    eye  = torch.eye(B, dtype=torch.bool, device=z.device)

    within_mask  = same & ~eye
    between_mask = ~same

    within_mean  = D2[within_mask].mean() if within_mask.any() else torch.tensor(0., device=z.device)
    between_mean = D2[between_mask].mean() if between_mask.any() else torch.tensor(0., device=z.device)
    return within_mean, between_mean

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



def train_unsupervised(model, trainloader, device, epochs=5):
    """
    Trains an unsupervised model (e.g., autoencoder) with real-time loss visualization.
    Saves model weights, batch-specific losses, and epoch-level aggregated losses.
    """
    # Define the loss function
    criterion = nn.MSELoss()  # Mean Squared Error loss for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    model.train()

    # Create directories to store training results
    results_dir = "../loss"
    weights_dir = "../net_weights/unsup"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Initialize the data structure to store loss information
    batch_loss_values = []

    # Initialize real-time plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Unsupervised Training Loss")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    loss_line, = ax.plot([], [], label="Loss", color="blue")
    ax.legend()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}/{epochs}")
        running_loss = 0.0
        epoch_losses = []  # Store losses for the current epoch

        for batch_idx, (images, labels) in enumerate(trainloader, start=1):
            # Transfer images to the device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # # Forward pass
            # outputs = model(images)
            # loss = criterion(outputs, images)
            #
            # # Backward pass and optimization
            # loss.backward()
            # optimizer.step()

            # Use penalty to ensure within = between in latent (encoder) space
            # Forward
            eps = 1e-8

            # forward
            z = model.encoder(images)
            # ----- inside the training loop, after z = model.encoder(images) -----
            # Reconstruction
            x_hat = model.decoder(z)
            recon_loss = criterion(x_hat, images)

            # SAME METRIC as we’ll use everywhere: *squared* Euclidean on unit-norm z
            D2 = euclidean_distance_matrix(z, normalize=True)  # this returns *squared* distances (no sqrt)

            B = D2.size(0)
            same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
            eye = torch.eye(B, dtype=torch.bool, device=z.device)

            within_mean = D2[same & ~eye].mean()
            between_mean = D2[~same].mean()

            # scale-invariant, well-conditioned penalty
            wb_penalty = ((within_mean - between_mean) / (between_mean.detach() + 1e-8)) ** 2

            # (optional) drop moment penalty for now; add later with its own λ if needed
            #α = 0.2  # try 0.2–0.5 to make it *bite*
            # curriculum over epochs (outside batch loop)
            if epoch < 3:
                α = 0  # pure reconstruction warmup
            elif epoch < 6:
                α = 0.2  # introduce constraint
            else:
                α = 0.2  # make it stronger

            λ_wb = α * (recon_loss.detach() / (wb_penalty.detach() + 1e-8))

            loss = recon_loss #+ λ_wb * wb_penalty

            # # --- DEBUG: check that the penalty is actually biting ---
            # if batch_idx % 7 == 0:  # print every 50 batches (adjust as you like)
            #     print(f"[dbg] epoch={epoch} batch={batch_idx} "
            #           f"recon={recon_loss.item():.6f} "
            #           f"within2={within_mean.item():.6f} "
            #           f"between2={between_mean.item():.6f} "
            #           f"gap2={(within_mean - between_mean).item():.6f} "
            #           f"wb_pen={wb_penalty.item():.6e} "
            #           f"lambda_wb={λ_wb.item():.6e}")

            loss.backward()
            optimizer.step()

            # Record batch loss
            batch_loss = loss.item()
            running_loss += batch_loss
            batch_loss_values.append(batch_loss)
            epoch_losses.append(batch_loss)

            # Save batch loss
            batch_loss_file = os.path.join(
                results_dir, f"unsup_loss_epoch_{epoch}_batch_{batch_idx}.npy"
            )
            np.save(batch_loss_file, np.array(batch_loss))
            print(f"Saved batch loss to: {batch_loss_file}")

            # Save model weights for the current batch
            batch_weight_path = os.path.join(
                weights_dir, f"unsup_net_weights_epoch_{epoch}_batch_{batch_idx}.pth"
            )
            torch.save(model.state_dict(), batch_weight_path)

            # Update real-time plot
            loss_line.set_xdata(range(1, len(batch_loss_values) + 1))
            loss_line.set_ydata(batch_loss_values)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.1)  # Brief pause for plot updates

            # Print batch information
            print(
                f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(trainloader)}], "
                f"Loss: {batch_loss:.4f}"
            )

        # Calculate and save epoch-level statistics
        avg_loss = running_loss / len(trainloader)
        epoch_loss_file = os.path.join(results_dir, f"unsup_losses_epoch_{epoch}.npy")
        np.save(epoch_loss_file, np.array(epoch_losses))
        print(f"Saved epoch-level loss to: {epoch_loss_file}")
        print(f"Epoch [{epoch}/{epochs}] completed with Avg Loss: {avg_loss:.4f}")

    # Save all batch losses collected during training
    all_loss_file = os.path.join(results_dir, "unsup_all_batch_losses.npy")
    np.save(all_loss_file, np.array(batch_loss_values))
    print(f"All batch losses saved to: {all_loss_file}")

    # Finalize the plot
    plt.ioff()  # Disable interactive mode
    plt.close(fig)  # Explicitly close the specific figure
    print("Unsupervised training complete.")




if __name__ == "__main__":
    # Path to your Excel file
    # Define the relative path
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")

    # Load the data
    #trainloader, valloader, testloader = load_gabor_data(excel_file,batch_size=64)

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
    # Initialize the autoencoder model
    unsup_net = Net()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the model
    train_unsupervised(unsup_net, trainloader, device, epochs=15)