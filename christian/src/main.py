import torch

from train_unsupervised import train_unsupervised
from dataset import load_gabor_data  # Importing the data loading function from dataset.py
#from Net import Net,SupervisedNet  # Import the autoencoder model from Net.py
from Net import Net,SupervisedNet  # Import the autoencoder model from Net.py
#from Mlp_Net import Net,SupervisedNet  # Import the autoencoder model from Net.py

from train_unsupervised import train_unsupervised
from train_supervised import train_supervised
import os
import numpy as np
import matplotlib
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring_2 import load_shape_deform_data

from shape_deform_dataset_v19_studio_full_main import ShapeDeformDataset,set_studio_from_main
from torch.utils.data import Dataset, DataLoader, random_split


matplotlib.use("TkAgg")  # Replace with a backend that supports interactivity

if __name__ == "__main__":

    # Path to your Excel file
    # Define the relative path
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")

    # Load the data
    #, valloader, testloader = load_gabor_data(excel_file,batch_size=64)
    #trainloader, valloader, testloader, full_dataset = load_many_arcs_data(m_arcs_per_class=5,gap_frac=0.3)
    #trainloader, valloader, testloader, full_dataset = load_image_ring_data(m_arcs_per_class=16,gap_frac=0.3)
    # ---- Configurable parameters ----
    # ---- Configurable parameters ----
    # =====================
    # Hard-coded settings
    # =====================
    # Counts
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

    # Initialize the autoencoder model
    unsup_net = Net()
    # Check if GPU is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unsup_net.to(device)

    # Train the unsupervised model
    train_unsupervised(unsup_net, trainloader, device, epochs=20)

    # Load the last available weights based on the count
    # Path to the weights folder
    weights_dir = "../net_weights/unsup/"
    # Count the number of files that match the pattern
    file_count = len([f for f in os.listdir(weights_dir) if f.startswith("unsup_net_weights_") and f.endswith(".pth")])

    # Find all matching weight files
    import re
    weight_files = [f for f in os.listdir(weights_dir) if re.match(r"unsup_net_weights_epoch_\d+_batch_\d+\.pth", f)]
    if weight_files:
        # Extract (epoch, batch) pairs and find the highest epoch and batch
        epoch_batch_pairs = [
            (int(re.search(r"epoch_(\d+)", f).group(1)), int(re.search(r"batch_(\d+)", f).group(1)))
            for f in weight_files
        ]
        latest_epoch, latest_batch = max(epoch_batch_pairs, key=lambda x: (x[0], x[1]))  # Max by epoch, and then by batch
        weight_path = os.path.join(weights_dir, f"unsup_net_weights_epoch_{latest_epoch}_batch_{latest_batch}.pth")
        unsup_net.load_state_dict(torch.load(weight_path))
        print(f"Loaded weights from: {weight_path}")
    else:
        raise FileNotFoundError("No weight files found in the folder.")

    # Train the supervised model
    # # Initialize the supervised model using the encoder from the trained autoencoder
    sup_net = SupervisedNet(unsup_net)
    sup_net.to(device)
    train_supervised(sup_net, trainloader, device, epochs=10)
