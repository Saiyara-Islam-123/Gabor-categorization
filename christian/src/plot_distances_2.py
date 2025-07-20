import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import Isomap
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("TkAgg")  # Replace with a backend that supports interactivity


def plot_distances_and_losses(num_unsup_epochs, num_sup_epochs, results_dir="../epochs_results", activations_dir="../epochs_results"):
    """
    Plots the evolution of distances and losses for unsupervised and supervised training epochs dynamically.
    Additionally, shows a 2D scatter plot of activations using MDS in a separate figure.

    Args:
        num_unsup_epochs: Number of unsupervised training epochs.
        num_sup_epochs: Number of supervised training epochs.
        results_dir: Directory containing distance and loss results.
        activations_dir: Directory containing activations saved per epoch.
    """
    # Paths to the saved results
    unsup_within_file_cat0 = os.path.join(results_dir, "unsup_within_distances_cat0.npy")
    unsup_within_file_cat1 = os.path.join(results_dir, "unsup_within_distances_cat1.npy")
    unsup_within_file = os.path.join(results_dir, "unsup_within_distances.npy")
    unsup_between_file = os.path.join(results_dir, "unsup_between_distances.npy")
    sup_within_file_cat0 = os.path.join(results_dir, "sup_within_distances_cat0.npy")
    sup_within_file_cat1 = os.path.join(results_dir, "sup_within_distances_cat1.npy")
    sup_within_file = os.path.join(results_dir, "sup_within_distances.npy")
    sup_between_file = os.path.join(results_dir, "sup_between_distances.npy")
    unsup_loss_file = os.path.join(results_dir, "unsup_epoch_losses.npy")
    sup_loss_file = os.path.join(results_dir, "sup_epoch_losses.npy")

    # Load the saved distances and losses
    print("Loading distances and losses from files...")
    unsup_within_cat0 = np.load(unsup_within_file_cat0)
    unsup_within_cat1 = np.load(unsup_within_file_cat1)
    unsup_within = np.load(unsup_within_file)
    unsup_between = np.load(unsup_between_file)
    sup_within_cat0 = np.load(sup_within_file_cat0)
    sup_within_cat1 = np.load(sup_within_file_cat1)
    sup_within = np.load(sup_within_file)
    sup_between = np.load(sup_between_file)
    unsup_losses = np.load(unsup_loss_file)
    sup_losses = np.load(sup_loss_file)

    # Combine results from unsupervised and supervised training
    within_distances = np.concatenate((unsup_within, sup_within))
    within_distances_cat0 = np.concatenate((unsup_within_cat0, sup_within_cat0))
    within_distances_cat1 = np.concatenate((unsup_within_cat1, sup_within_cat1))

    between_distances = np.concatenate((unsup_between, sup_between))
    losses = np.concatenate((unsup_losses, sup_losses))

    # Total number of epochs
    total_epochs = num_unsup_epochs + num_sup_epochs
    epochs = np.arange(1, total_epochs + 1)

    # Setup the main figure for distances and losses
    plt.ion()
    fig, (ax_loss, ax_dist) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Loss subplot
    ax_loss.set_title("Evolution of Losses")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xlim(1, total_epochs)
    loss_unsup_line, = ax_loss.plot([], [], label="Unsupervised Loss", color="purple", marker="o")
    loss_sup_line, = ax_loss.plot([], [], label="Supervised Loss", color="orange", marker="s")
    ax_loss.axvline(num_unsup_epochs + 0.5, color="red", linestyle="--", label="Supervised Start")
    ax_loss.legend()
    ax_loss.grid()

    # Distance subplot
    ax_dist.set_title("Evolution of Cosine Distances")
    ax_dist.set_xlabel("Epochs")
    ax_dist.set_ylabel("Cosine Distance")
    #within_unsup_line, = ax_dist.plot([], [], label="Unsupervised Within-Category", color="green", marker="o")
    within_unsup_line_cat0, = ax_dist.plot([], [], label="Unsupervised Within-Category_0", color="green", marker="o")
    within_unsup_line_cat1, = ax_dist.plot([], [], label="Unsupervised Within-Category_1", color="lime", marker="o")

    between_unsup_line, = ax_dist.plot([], [], label="Unsupervised Between-Category", color="blue", marker="o")
    within_sup_line_cat0, = ax_dist.plot([], [], label="Supervised Within-Category_cat0", color="green", marker="s", linestyle="--")
    within_sup_line_cat1, = ax_dist.plot([], [], label="Supervised Within-Category_cat1", color="lime", marker="s", linestyle="--")
    #within_sup_line, = ax_dist.plot([], [], label="Supervised Within-Category", color="green", marker="s", linestyle="--")



    between_sup_line, = ax_dist.plot([], [], label="Supervised Between-Category", color="blue", marker="s", linestyle="--")
    ax_dist.axvline(num_unsup_epochs + 0.5, color="red", linestyle="--", label="Supervised Start")
    ax_dist.set_xlim(1, total_epochs)
    ax_dist.legend()
    ax_dist.grid()

    # Separate figure for 2D scatter plot
    scatter_fig, scatter_ax = plt.subplots(figsize=(8, 6))
    scatter_ax.set_title("2D Visualization of Activations")
    scatter_ax.set_xlabel("Dimension 1")
    scatter_ax.set_ylabel("Dimension 2")
    scatter_ax.grid()

    # Real-time updates
    for i in range(total_epochs):
        current_epoch = i + 1

        # Update loss values
        if current_epoch <= num_unsup_epochs:
            loss_unsup_line.set_data(epochs[:current_epoch], losses[:current_epoch])
        else:
            loss_sup_line.set_data(epochs[num_unsup_epochs:current_epoch], losses[num_unsup_epochs:current_epoch])

        # Update distance values
        if current_epoch <= num_unsup_epochs:
            within_unsup_line_cat0.set_data(epochs[:current_epoch], within_distances_cat0[:current_epoch])
            within_unsup_line_cat1.set_data(epochs[:current_epoch], within_distances_cat1[:current_epoch])
            #within_unsup_line.set_data(epochs[:current_epoch], within_distances[:current_epoch])
            between_unsup_line.set_data(epochs[:current_epoch], between_distances[:current_epoch])
        else:
            within_sup_line_cat0.set_data(epochs[num_unsup_epochs:current_epoch], within_distances_cat0[num_unsup_epochs:current_epoch])
            within_sup_line_cat1.set_data(epochs[num_unsup_epochs:current_epoch], within_distances_cat1[num_unsup_epochs:current_epoch])
            #within_sup_line.set_data(epochs[num_unsup_epochs:current_epoch], within_distances[num_unsup_epochs:current_epoch])
            between_sup_line.set_data(epochs[num_unsup_epochs:current_epoch], between_distances[num_unsup_epochs:current_epoch])

        # Dynamically adjust axis limits
        max_loss = losses[:current_epoch].max()
        ax_loss.set_ylim(0, max(max_loss * 1.1, 0.1))

        current_max_dist = max(within_distances[:current_epoch].max(), between_distances[:current_epoch].max())
        ax_dist.set_ylim(0, current_max_dist * 1.1)

        # Update scatter plot (2D MDS)
        activations_file = os.path.join(activations_dir, f"unsup_activations_epoch_{current_epoch - 1}.npy") if current_epoch <= num_unsup_epochs else \
                           os.path.join(activations_dir, f"sup_activations_epoch_{current_epoch - num_unsup_epochs - 1}.npy")
        labels_file = os.path.join(activations_dir, f"unsup_labels_epoch_{current_epoch - 1}.npy") if current_epoch <= num_unsup_epochs else \
                      os.path.join(activations_dir, f"sup_labels_epoch_{current_epoch - num_unsup_epochs - 1}.npy")
        if os.path.exists(activations_file) and os.path.exists(labels_file):
            activations = np.load(activations_file)
            labels = np.load(labels_file)
            reduced = MDS(n_components=2, random_state=0)
            #reduced = PCA(n_components=2, random_state=0)
            #reduced = TSNE(n_components=2, random_state=0)
            #reduced = umap.UMAP(n_components=2, random_state=0)
            #reduced = LocallyLinearEmbedding(n_components=3, random_state=0)
            #reduced = Isomap(n_components=2)
            #reduced = FactorAnalysis(n_components=2,random_state=0)

            #scaler = StandardScaler()
            #scatter_data = scaler.fit_transform(activations)

            scatter_data = reduced.fit_transform(activations)

            scatter_ax.clear()
            # Map colors based on labels
            colors = ["green" if label == 0 else "lime" for label in labels]

            scatter_ax.scatter(scatter_data[:, 0], scatter_data[:, 1], c=colors, cmap='tab10', s=10, alpha=0.8)
            scatter_ax.set_title("2D Visualization of Activations")
            scatter_ax.set_xlabel("Dimension 1")
            scatter_ax.set_ylabel("Dimension 2")
            scatter_ax.grid()
            scatter_fig.canvas.draw()

        # Redraw the main figure
        fig.canvas.draw()
        plt.pause(0.2)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    num_unsup_epochs = 15
    num_sup_epochs = 15
    plot_distances_and_losses(num_unsup_epochs, num_sup_epochs)