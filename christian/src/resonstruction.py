import torch
import os
import matplotlib.pyplot as plt
from rfc3987 import upatterns_no_names

from dataset import load_gabor_data  # Importing the data loader
from Net import Net
from Transformer import Transformer


def load_latest_weights(model, weights_dir):
    """
    Loads the latest weights for the given model from the specified directory. The
    function looks for weight files following a naming pattern 'unsup_net_weights_<epoch>.pth',
    where <epoch> is an integer representing the epoch number. It sorts these files
    by their epoch numbers and loads the latest one into the model.

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

    # Find all files in the directory
    weight_files = [
        f for f in os.listdir(weights_dir)
        if f.startswith("unsup_net_weights_") and f.endswith(".pth")
    ]
    if not weight_files:
        raise FileNotFoundError(f"No weights found in directory '{weights_dir}'.")

    # Sort files by epoch number
    weight_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))  # Extract epoch number
    latest_weights = weight_files[-1]  # Take the last file (latest epoch)
    print(latest_weights)
    # Load weights
    latest_weights_path = os.path.join(weights_dir, latest_weights)
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
    excel_file = "categorisation 4000.xlsx"
    weights_dir = "../net_weights/unsup/"

    # Load the data
    _, _, testloader = load_gabor_data(excel_file, batch_size=32)  # Only need the test loader

    # Initialize the model
    unsup_net = Transformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    unsup_net.to(device)

    # Load the latest weights
    weight_path = "../net_weights/Transformer_unsup/unsup_weights_ lr= 0.001 2 32.pth"
    unsup_net.load_state_dict(torch.load(weight_path))

    # Test reconstruction with the model
    reconstruction(unsup_net, testloader, device)
