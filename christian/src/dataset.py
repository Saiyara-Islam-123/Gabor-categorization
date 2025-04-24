import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os


# Custom dataset class for loading images and labels from Excel
class GaborDataset(Dataset):
    def __init__(self, excel_file, transform=None):
        """
        Initializes an instance of a class that processes data from an Excel file
        and optionally applies a transformation function.

        :param excel_file: Path to the Excel file that needs to be read. The file
            should be in a format supported by pandas.read_excel.
        :type excel_file: str
        :param transform: Optional callable function to apply custom transformations
            on the data after reading it from the file.
            Defaults to None if no transformation is provided.
        :type transform: Callable, optional
        """
        self.data = pd.read_excel(excel_file)  # Read the Excel file
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset (image and label).
        """
        img_path = self.data.iloc[idx, 0]  # Assuming image paths are in the first column
        label = self.data.iloc[idx, 1]  # Assuming labels are in the second column

        # Load the image from the file path
        # Define the base path relative to the src folder (e.g., go up one directory)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Combine with the relative path from img_path
        img_path = os.path.join(base_dir, img_path)

        image = Image.open(img_path).convert('RGB')  # Convert image to RGB mode

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Convert the label to a tensor (if it's categorical, convert it to an integer first)
        label = torch.tensor(0 if label == 'k' else 1)  # Assuming 'k' -> 0, 'l' -> 1

        return image, label


# Data loader function
def load_gabor_data(excel_file, batch_size=64):
    # Define a transform to preprocess the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor()  # Converts image to tensor and scales to [0, 1]
    ])

    # Load the dataset
    dataset = GaborDataset(excel_file, transform=transform)

    # Split the dataset into training (80%), validation (10%), and test (10%)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader


# Helper function to plot images
def plot_images(images, labels):
    """Plot a grid of images and their corresponding labels."""
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))  # Convert (C, H, W) tensor to (H, W, C) for plotting
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Path to your Excel file
    # Define the relative path
    excel_file = os.path.join(os.path.expanduser("~"), "Gabor-categorization", "christian", "experimentFiles","categorisation.xlsx")

    # Load the data
    trainloader, valloader, testloader = load_gabor_data(excel_file)

    # Get a batch of images and labels from the training set
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Plot the images
    plot_images(images, labels)

    # Print batch shapes to confirm
    print(f"Batch of images shape: {images.shape}")
    print(f"Batch of labels shape: {labels.shape}")
