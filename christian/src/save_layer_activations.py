import os
import torch
import numpy as np
from dataset import load_gabor_data
from Net import Net, SupervisedNet
#from Mlp_Net import Net, SupervisedNet
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring_2 import load_shape_deform_data


def evaluate_and_save_batches(model, trainloader, device, weight_dir, num_epochs, num_batches, save_prefix):
    """
    Evaluates a model across multiple epochs and batches using provided training data and saves
    the layer activations for each epoch and batch.

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
    :param num_batches: The number of batches to evaluate within each epoch.
    :type num_batches: int
    :param save_prefix: Prefix for the filenames of the saved activations.
    :type save_prefix: str
    :return: None
    """
    # Results directory
    results_dir = "../activations"
    os.makedirs(results_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for batch in range(1, num_batches + 1):
            print(f"Processing Epoch {epoch}/{num_epochs}, Batch {batch}/{num_batches}...")

            # Load model weights for the epoch
            if weight_dir.endswith("unsup"):
                weight_path = os.path.join(weight_dir, f"unsup_net_weights_epoch_{epoch}_batch_{batch}.pth")
            else:
                weight_path = os.path.join(weight_dir, f"sup_net_weights_epoch_{epoch}_batch_{batch}.pth")

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
            epoch_batch_activations = torch.cat(all_activations, dim=0)
            epoch_batch_labels = torch.cat(all_labels, dim=0)

            # Save activations and labels for the current epoch and batch

            activation_file = os.path.join(
                results_dir, f"{save_prefix}_activations_epoch_{epoch}_batch_{batch}.npy"
            )
            labels_file = os.path.join(
                results_dir, f"{save_prefix}_labels_epoch_{epoch}_batch_{batch}.npy"
            )
            np.save(activation_file, epoch_batch_activations.numpy())
            np.save(labels_file, epoch_batch_labels.numpy())

            print(f"Saved activations for epoch {epoch}, batch {batch}")



if __name__ == "__main__":
    # Paths to weight directories
    unsup_weight_dir = os.path.abspath("../net_weights/unsup")
    sup_weight_dir = os.path.abspath("../net_weights/sup")

    # Number of epochs and batches
    num_unsup_epochs = 20
    num_unsup_batches = 7  # Adjust based on training configuration for unsupervised learning
    num_sup_epochs = 10
    num_sup_batches = 7    # Adjust batches for supervised learning

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

    # Save activations for unsupervised model
    print("Processing unsupervised model activations...")
    unsup_net = Net()
    unsup_net.to(device)
    evaluate_and_save_batches(unsup_net, trainloader, device, unsup_weight_dir, num_unsup_epochs, num_unsup_batches, "unsup")

    # Save activations for supervised model
    print("Processing supervised model activations...")
    sup_net = SupervisedNet(unsup_net)  # Use unsupervised encoder weights
    sup_net.to(device)
    evaluate_and_save_batches(sup_net, trainloader, device, sup_weight_dir, num_sup_epochs, num_sup_batches, "sup")
