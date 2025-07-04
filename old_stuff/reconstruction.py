import torch
import os
import matplotlib.pyplot as plt
from dataset import load_gabor_data  # Importing the data loader


def reconstruction(model, testloader):
    """
    Test the model's reconstruction ability on the first batch of test images.
    Displays the first 4 input samples and their corresponding reconstructions.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Get the first batch of test images
        for images, _ in testloader:

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
            break