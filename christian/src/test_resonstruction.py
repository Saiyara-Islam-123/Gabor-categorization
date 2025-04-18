import torch
import os
import matplotlib.pyplot as plt
from dataset import load_gabor_data  # Importing the data loader
from Net import Net


def load_latest_weights(model, weights_dir):
    """
    Load the latest weights from the directory. Assumes files are named with epoch numbers, e.g., `unsup_net_weights_epoch_15.pth`.
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


def test_reconstruction(model, testloader, device):
    """
    Test the model's reconstruction ability on the first batch of test images.
    Displays the first 4 input samples and their corresponding reconstructions.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get the first batch of test images
        for images, _ in testloader:
            images = images.to(device)
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
    _, _, testloader = load_gabor_data(excel_file, batch_size=64)  # Only need the test loader

    # Initialize the model
    unsup_net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    unsup_net.to(device)

    # Load the latest weights
    load_latest_weights(unsup_net, weights_dir)

    # Test reconstruction with the model
    test_reconstruction(unsup_net, testloader, device)
