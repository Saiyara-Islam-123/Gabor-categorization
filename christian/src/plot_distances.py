import os
import numpy as np
import matplotlib.pyplot as plt


def plot_distances_and_losses(num_unsup_epochs, num_sup_epochs, results_dir="../epochs_results"):
    """
    Fake real-time plot of the evolution of distances and losses for unsupervised and supervised epochs.
    Dynamically updates distance and loss subplots, showing 40 total ticks (20 unsupervised + 20 supervised),
    and resets epoch labels every 20 epochs.
    """
    # Paths to the saved results
    unsup_within_file = os.path.join(results_dir, "unsup_within_distances.npy")
    unsup_between_file = os.path.join(results_dir, "unsup_between_distances.npy")
    sup_within_file = os.path.join(results_dir, "sup_within_distances.npy")
    sup_between_file = os.path.join(results_dir, "sup_between_distances.npy")
    unsup_loss_file = os.path.join(results_dir, "unsup_epoch_losses.npy")
    sup_loss_file = os.path.join(results_dir, "sup_epoch_losses.npy")

    # Load the saved distances and losses
    print("Loading distances and losses from files...")
    unsup_within = np.load(unsup_within_file)
    unsup_between = np.load(unsup_between_file)
    sup_within = np.load(sup_within_file)
    sup_between = np.load(sup_between_file)
    unsup_losses = np.load(unsup_loss_file)
    sup_losses = np.load(sup_loss_file)

    # Combine results from unsupervised and supervised
    within_distances = list(unsup_within) + list(sup_within)
    between_distances = list(unsup_between) + list(sup_between)
    losses = list(unsup_losses) + list(sup_losses)

    # Total number of epochs and x-axis settings
    total_epochs = num_unsup_epochs + num_sup_epochs
    epochs = list(range(1, total_epochs + 1))  # Absolute epoch numbers (1 to 40)
    epoch_labels = [(epoch - 1) % 20 + 1 for epoch in epochs]  # Labels reset every 20 epochs (1-20)

    # Initialize plots
    plt.ion()
    fig, (ax_loss, ax_dist) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    # Distance subplot
    within_line_unsup, = ax_dist.plot([], [], label="Unsupervised Within-Category Distance", color="green", marker="o")
    between_line_unsup, = ax_dist.plot([], [], label="Unsupervised Between-Category Distance", color="blue", marker="o")
    within_line_sup, = ax_dist.plot([], [], label="Supervised Within-Category Distance", color="darkgreen", marker="s",
                                    linestyle="--")
    between_line_sup, = ax_dist.plot([], [], label="Supervised Between-Category Distance", color="darkblue", marker="s",
                                     linestyle="--")
    ax_dist.axvline(num_unsup_epochs + 0.5, color="red", linestyle="--", label="Supervised Learning Start")
    ax_dist.set_xlim(1, total_epochs)  # Full range of epochs (1 to 40)
    ax_dist.set_ylim(-0.1, 1)  # Allow slightly negative y-axis values
    ax_dist.set_ylabel("Average Cosine Distance")
    ax_dist.set_title("Evolution of Cosine Distances")
    ax_dist.legend()
    ax_dist.grid()

    # Loss subplot
    loss_unsup_line, = ax_loss.plot([], [], label="Unsupervised Loss", color="purple", marker="o", linestyle="-")
    loss_sup_line, = ax_loss.plot([], [], label="Supervised Loss", color="orange", marker="s", linestyle="--")
    ax_loss.axvline(num_unsup_epochs + 0.5, color="red", linestyle="--", label="Supervised Learning Start")
    ax_loss.set_xlim(1, total_epochs)  # Match full range
    ax_loss.set_ylim(0, max(unsup_losses[0], sup_losses[0], 1))  # Start with loss range
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Evolution of Losses")
    ax_loss.legend()
    ax_loss.grid()

    # Real-time updates
    for i in range(total_epochs):
        current_epoch = epochs[i]
        current_label = epoch_labels[i]

        # Update Loss Plot
        max_loss = max(losses[:i + 1])  # Adjust y-axis for loss plot
        ax_loss.set_ylim(0, max(0.1, max_loss * 1.1))
        if i < num_unsup_epochs:
            loss_unsup_line.set_xdata(epochs[:i + 1])
            loss_unsup_line.set_ydata(losses[:i + 1])
        else:
            loss_sup_line.set_xdata(epochs[num_unsup_epochs:i + 1])
            loss_sup_line.set_ydata(losses[num_unsup_epochs:i + 1])

        # Update Distance Plot
        max_within = max(within_distances[:i + 1])
        max_between = max(between_distances[:i + 1])
        current_max = max(max_within, max_between)
        current_min = min(0, min(within_distances[:i + 1] + between_distances[:i + 1]))
        ax_dist.set_ylim(current_min - 0.1, max(0.1, current_max * 1.1))
        if i < num_unsup_epochs:
            within_line_unsup.set_xdata(epochs[:i + 1])
            within_line_unsup.set_ydata(within_distances[:i + 1])
            between_line_unsup.set_xdata(epochs[:i + 1])
            between_line_unsup.set_ydata(between_distances[:i + 1])
        else:
            within_line_sup.set_xdata(epochs[num_unsup_epochs:i + 1])
            within_line_sup.set_ydata(within_distances[num_unsup_epochs:i + 1])
            between_line_sup.set_xdata(epochs[num_unsup_epochs:i + 1])
            between_line_sup.set_ydata(between_distances[num_unsup_epochs:i + 1])

        # Update X-axis Ticks and Labels
        ax_dist.set_xticks(epochs)
        ax_dist.set_xticklabels(epoch_labels)
        ax_loss.set_xticks(epochs)
        ax_loss.set_xticklabels(epoch_labels)

        # Redraw plots
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Pause to simulate real-time plotting
        plt.pause(0.2)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    num_unsup_epochs = 20
    num_sup_epochs = 20

    plot_distances_and_losses(num_unsup_epochs, num_sup_epochs)
