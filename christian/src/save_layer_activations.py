import os
import torch
import numpy as np
from dataset import load_gabor_data
from Net import Net, SupervisedNet
#from Mlp_Net import Net, SupervisedNet
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring_2 import load_shape_deform_data
from shape_deform_dataset_v19_studio_full_main import ShapeDeformDataset,set_studio_from_main
from torch.utils.data import Dataset, DataLoader, random_split


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
    nA = 500  # class 0
    nB = nA  # class 1

    # Image/output settings
    image_size = 128
    intensity = 1.0
    bg = 0.0
    norm = "max"  # "max" | "l2" | "none"
    seed = 42
    batch_size = 64

    # Actions
    PRINT_STATS = False
    SHOW_EXAMPLES = True
    export_gabors = True

    # Studio preference (all sliders/knobs exposed exactly like older mains)
    set_studio_from_main(
        m_phase=3, m_amp=3, gap=0.25, phase_deg=0.0,
        which_arc_phi=0, pos_phi=0.5, which_arc_amp=0, pos_amp=0.5,
        R=5.0, profile="absolute", sharp=0.4, amp_min=None, amp_max=None,
        k_max=8, m_freq=3, gap_freq=0.50,
        phase_src="None (ring)", amp_src="None (ring)", freq_src="None (ring)",
        k1=3, a1=0.9, phi1_deg=45.0, phase_mode1="signed_absolute", sphi1=0.25, Kphi1=10.0,
        amp_mode1="relative", sA1=0.25, KA1=10.0,
        k2=5, a2=0.6, phi2_deg=0.0, phase_mode2="signed_absolute", sphi2=0.25, Kphi2=10.0,
        amp_mode2="relative", sA2=0.25, KA2=10.0,
        tinys=[(8, 0.0, 0.0, 0.5), (14, 0.0, 0.0, 0.8), (2, 0.0, 0.0, 0.3)]
    )

    # Build dataset by asking Studio to generate batches until we have nA/nB
    ds = ShapeDeformDataset(nA=nA, nB=nB,
                            image_size=image_size,
                            intensity=intensity, bg=bg, norm=norm,
                            seed=seed, batch=8)

    # Split + loaders
    total = len(ds)
    val_sz = max(1, total // 5)  # 20%
    test_sz = max(1, total // 5)  # 20%
    train_sz = total - val_sz - test_sz
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [train_sz, val_sz, test_sz], generator=g)
    mk_loader = lambda d: DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=False)
    trainloader, val_loader, test_loader = mk_loader(train_ds), mk_loader(val_ds), mk_loader(test_ds)

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
