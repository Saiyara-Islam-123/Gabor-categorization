import torch
import os
import matplotlib.pyplot as plt
from dataset import load_gabor_data  # Importing the data loader
from Net import Net
#from Mlp_Net import Net
from shapes_paremetrized_with_ring import load_shape_deform_data
import numpy as np

def load_latest_weights(model, weights_dir):
    """
    Loads the latest weights for the given model from the specified directory.
    The function considers both epoch and batch when identifying the most recent
    weight file.

    :param model: The PyTorch model instance whose weights are to be loaded.
    :type model: torch.nn.Module
    :param weights_dir: The directory path where the weight files are stored.
    :type weights_dir: str
    :return: None
    :rtype: None
    :raises FileNotFoundError: If the weights directory does not exist or if no
        valid weight files are found in the directory.
    """
    if not os.path.exists(weights_dir):
        raise FileNotFoundError(f"Weights directory '{weights_dir}' does not exist.")

    # Find all files matching the naming pattern for weights
    import re
    weight_files = [
        f for f in os.listdir(weights_dir) if re.match(r"unsup_net_weights_epoch_\d+_batch_\d+\.pth", f)
    ]
    if not weight_files:
        raise FileNotFoundError(f"No weights found in directory '{weights_dir}'.")

    # Extract (epoch, batch) pairs
    epoch_batch_pairs = [
        (int(re.search(r"epoch_(\d+)", f).group(1)), int(re.search(r"batch_(\d+)", f).group(1)))
        for f in weight_files
    ]

    # Find the file with the latest epoch and batch
    latest_epoch, latest_batch = max(epoch_batch_pairs, key=lambda x: (x[0], x[1]))
    latest_weights_file = f"unsup_net_weights_epoch_{latest_epoch}_batch_{latest_batch}.pth"

    # Load the latest weights
    latest_weights_path = os.path.join(weights_dir, latest_weights_file)
    model.load_state_dict(torch.load(latest_weights_path))
    print(f"Loaded weights from: {latest_weights_path}")



def reconstruction(model, testloader, device):
    """
    Test the model's reconstruction ability on the first batch of test images.
    Displays the first 4 input samples and their corresponding reconstructions.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get the first batch of test images
        for images, _ in testloader:
            images = images.to(device)

           # Add single horizontal stripe across the middle
           #  stripe_width = 2
           #  middle = images.shape[2] // 2
           #  stripe_start = middle - stripe_width // 2
           #  stripe_end = middle + stripe_width // 2
           #  images[:, :, stripe_start:stripe_end, :] = 1

            outputs = model(images)  # Reconstructed images

            # Display the first 4 images and their reconstructions
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            fig.suptitle("Reconstruction Results", fontsize=16)

            for i in range(4):
                # Handle images with 3 channels or 1 channel
                input_image = images[i].cpu().detach().numpy()  # Convert to NumPy
                reconstructed_image = outputs[i].cpu().detach().numpy()  # Convert to NumPy

                if input_image.shape[0] == 3:  # 3-channel image (e.g., RGB)
                    input_image = input_image.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
                    reconstructed_image = reconstructed_image.transpose(1, 2, 0)  # Same for reconstruction
                else:  # 1-channel image (grayscale)
                    input_image = input_image.squeeze(0)  # Remove the channel dimension
                    reconstructed_image = reconstructed_image.squeeze(0)  # Remove channel dimension

                # Display original image
                axes[0, i].imshow(input_image, cmap="gray")
                axes[0, i].axis("off")
                axes[0, i].set_title("Input")

                # Display reconstructed image
                axes[1, i].imshow(reconstructed_image, cmap="gray")
                axes[1, i].axis("off")
                axes[1, i].set_title("Reconstruction")

            plt.show()
            break  # Process only the first batch

if __name__ == "__main__":
    # Define paths and directories
    excel_file = os.path.join(
        os.path.expanduser("~"),
        "Gabor-categorization",
        "christian",
        "experimentFiles",
        "categorisation.xlsx"
    )
    weights_dir = "../net_weights/unsup/"

    # Load the data
    #_, _, testloader = load_gabor_data(excel_file, batch_size=64)  # Only need the test loader
    # ---- Configurable parameters ----
    m_arcs_per_class = 2
    gap_frac = (m_arcs_per_class * 0.1) / np.pi
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
    # Initialize the model
    unsup_net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    unsup_net.to(device)

    # Load the latest weights
    load_latest_weights(unsup_net, weights_dir)

    # Test reconstruction with the model
    reconstruction(unsup_net, testloader, device)
