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
    nA = 1000
    nB = nA
    m_arcs_per_class = 2
    gap_frac = 0.5
    phase_deg = 0.0

    # Amplitude ring params (independent)
    amp_m_arcs_per_class = 6
    amp_gap_frac = 0.2

    # Amplitude ring controls
    amp_ring_radius = 1  # smaller = subtler deformation
    amp_scale_a1 = 0.2
    amp_scale_a2 = 0.3

    difficulty_sharp = 0.15  # try 0.3–0.8; 0 = no sharpening difference

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
    )    # Initialize the autoencoder model
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
