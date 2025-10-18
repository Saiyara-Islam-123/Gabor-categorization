import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")      # show live figure
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

# NEW: import the function you provided
from rings_no_overlap import load_many_arcs_data
from images_of_rings_no_overlap import load_image_ring_data
from shapes_paremetrized_with_ring_2 import load_shape_deform_data
from shapes_paremetrized_with_ring_2 import plot_latent_scatter



def plot_and_record_video(num_unsup_epochs, num_sup_epochs, num_batches,
                          loss_dir="../loss", distances_dir="../distances",
                          activations_dir="../activations",
                          out_video="../plots/training_evolution.mp4",
                          dataset=None):  # NEW: pass dataset so we can draw input ring

    def load_list(prefix, name):
        return [np.mean(np.load(os.path.join(prefix, name.format(e, b))))
                for e in range(num_unsup_epochs + num_sup_epochs)
                for b in range(1, num_batches + 1)
                if os.path.exists(os.path.join(prefix, name.format(e, b)))]

    unsup_w0 = load_list(distances_dir, "unsup_within_distances_cat0_epoch_{}_batch_{}.npy")
    unsup_w1 = load_list(distances_dir, "unsup_within_distances_cat1_epoch_{}_batch_{}.npy")
    unsup_b  = load_list(distances_dir, "unsup_between_distances_epoch_{}_batch_{}.npy")
    unsup_l  = load_list(loss_dir,      "unsup_loss_epoch_{}_batch_{}.npy")

    sup_w0   = load_list(distances_dir, "sup_within_distances_cat0_epoch_{}_batch_{}.npy")
    sup_w1   = load_list(distances_dir, "sup_within_distances_cat1_epoch_{}_batch_{}.npy")
    sup_b    = load_list(distances_dir, "sup_between_distances_epoch_{}_batch_{}.npy")
    sup_l    = load_list(loss_dir,      "sup_loss_epoch_{}_batch_{}.npy")

    within0  = unsup_w0 + sup_w0
    within1  = unsup_w1 + sup_w1
    between  = unsup_b  + sup_b
    losses   = unsup_l  + sup_l  # (kept; not used for plotting lines now)

    # Load all activation snapshots first

    # Load all activation snapshots first
    acts_all, labels_all = [], []

    # Unsupervised epochs
    for e in range(num_unsup_epochs):
        for b in range(1, num_batches + 1):
            a_path = os.path.join(activations_dir, f"unsup_activations_epoch_{e}_batch_{b}.npy")
            l_path = os.path.join(activations_dir, f"unsup_labels_epoch_{e}_batch_{b}.npy")
            if os.path.exists(a_path) and os.path.exists(l_path):
                acts_all.append(np.load(a_path))
                labels_all.append(np.load(l_path))

    # Supervised epochs
    for e in range(num_sup_epochs):
        for b in range(1, num_batches + 1):
            a_path = os.path.join(activations_dir, f"sup_activations_epoch_{e}_batch_{b}.npy")
            l_path = os.path.join(activations_dir, f"sup_labels_epoch_{e}_batch_{b}.npy")
            if os.path.exists(a_path) and os.path.exists(l_path):
                acts_all.append(np.load(a_path))
                labels_all.append(np.load(l_path))

    # --- Force x-axis to exactly epochs * batches ---
    total_batches = (num_unsup_epochs + num_sup_epochs) * num_batches
    batches = np.arange(1, total_batches + 1)
    unsup_end = num_unsup_epochs * num_batches  # boundary index

    # ---------- figure layout ----------
    plt.ion()
    fig = plt.figure(figsize=(16, 8))   # wide figure: ring-above, latent-below on right
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[1, 0], sharex=ax_loss)

    # NEW: separate plot ABOVE the latent space plot (right column top/bottom)
    ax_ring    = fig.add_subplot(gs[0, 1])   # top-right: input ring
    ax_scatter = fig.add_subplot(gs[1, 1])   # bottom-right: latent space

    # --- FIXED scatter limits to keep square and constant ---
    ax_scatter.set_xlim(-7, 7)
    ax_scatter.set_ylim(-7, 7)
    ax_scatter.set_aspect('equal', adjustable='box')
    ax_scatter.set_title("Network space")
    ax_scatter.set_xlabel("Dim 1")
    ax_scatter.set_ylabel("Dim 2")
    scatter_points = ax_scatter.scatter([], [], s=16, alpha=0.8)

    # NEW: draw the original INPUT ring + points in the top-right subplot
    if dataset is not None:
        A, B = dataset.latents_A, dataset.latents_B
        ax_ring.scatter(A[:, 0], A[:, 1], s=10, color="green", alpha=0.7, label="Class 0")
        ax_ring.scatter(B[:, 0], B[:, 1], s=10, color="lime",  alpha=0.7, label="Class 1")
        ang = np.linspace(0, 2*np.pi, 400)
        ax_ring.plot(np.cos(ang), np.sin(ang),
                     lw=1.0, color="black", alpha=0.7)
        ax_ring.set_aspect("equal", adjustable="box")
        ax_ring.set_title("Input space")
        ax_ring.set_xlabel("x")
        ax_ring.set_ylabel("y")
        ax_ring.legend(loc="upper right")
        ax_ring.grid(True, ls="--", alpha=0.3)

        # ALSO: call your original function, exactly as requested
        plot_latent_scatter(dataset, title=f"Latent (x,y) – alternating arcs (m={m_arcs_per_class})")

    # Loss and distances
    ax_loss.set_title("Unsupervised and supervised error evolution")
    ax_loss.set_ylabel("Error")
    ax_loss.axvline(unsup_end + 0.5, color="red", linestyle="--", label="Supervised Learning Start")

    # separate loss lines + legend colors as requested
    loss_unsup_line, = ax_loss.plot([], [], color="green", linewidth=4.0, label="Unsupervised Error")
    loss_sup_line,   = ax_loss.plot([], [], color="orange",linewidth=4.0, label="Supervised Error")
    ax_loss.legend()
    ax_loss.grid()

    ax_dist.set_title("Within/Between distances")
    ax_dist.set_xlabel("Batch")
    ax_dist.set_ylabel("Distance")
    line_w0, = ax_dist.plot([], [], color="green",linewidth=4.0, label="Within Cat0")
    line_w1, = ax_dist.plot([], [], color="lime",linewidth=4.0,  label="Within Cat1")
    line_b , = ax_dist.plot([], [], color="blue", linewidth=4.0, label="Between")
    ax_dist.axvline(unsup_end + 0.5, color="red", linestyle="--", label="Supervised Learning Start")

    ax_dist.legend()
    ax_dist.grid()

    os.makedirs(os.path.dirname(out_video), exist_ok=True)
    writer = FFMpegWriter(fps=15, bitrate=1800)

    with writer.saving(fig, out_video, dpi=300):
        for frame in range(total_batches):
            cur = frame + 1

            # --- Update loss lines with the correct phase split ---
            if cur <= unsup_end:
                # Only unsupervised segment visible so far
                loss_unsup_line.set_data(batches[:cur], unsup_l[:cur])
                loss_sup_line.set_data([], [])
                current_loss_max = max(unsup_l[:cur]) if len(unsup_l) >= cur else (max(unsup_l) if unsup_l else 1.0)
            else:
                # Show full unsup, plus partial supervised
                loss_unsup_line.set_data(batches[:unsup_end], unsup_l[:unsup_end])
                sup_len = min(cur - unsup_end, len(sup_l))
                x_sup = batches[unsup_end:unsup_end + sup_len]
                loss_sup_line.set_data(x_sup, sup_l[:sup_len])
                current_loss_max = max(
                    (max(unsup_l) if unsup_l else 0.0),
                    (max(sup_l[:sup_len]) if sup_len > 0 else 0.0)
                )

            # Update distance lines (unchanged logic)
            upto = min(cur, len(within0))
            line_w0.set_data(batches[:upto], within0[:upto])
            line_w1.set_data(batches[:upto], within1[:upto])
            line_b .set_data(batches[:upto], between[:upto])

            # Axes limits exactly epochs * batches
            ax_loss.set_xlim(1, total_batches)
            ax_dist.set_xlim(1, total_batches)

            # Y-lims
            ax_loss.set_ylim(0, (current_loss_max if current_loss_max > 0 else 1.0) * 1.1)
            if upto > 0:
                ax_dist.set_ylim(0, max(max(within0[:upto]),
                                        max(within1[:upto]),
                                        max(between[:upto])) * 1.1)
            print(len(acts_all))
            # Update latent scatter (axes stay fixed at [-5,5])
            # --- inside the frame loop ---
            if frame < len(acts_all):
                reduced = PCA(n_components=2, random_state=0).fit_transform(acts_all[frame])
                lbls = labels_all[frame]
                cols = np.where(lbls == 0, "green", "lime")
                scatter_points.set_offsets(reduced)
                scatter_points.set_color(cols)

            fig.canvas.draw()
            plt.pause(0.01)
            writer.grab_frame()

    plt.ioff()
    plt.show()
    print(f"Video saved to {out_video}")


if __name__ == "__main__":
    # ---- Configurable parameters ----
    # ---- Configurable parameters ----
    nA = 1000
    nB = nA
    m_arcs_per_class = 2
    gap_frac = 0.5
    phase_deg = 0.0

    # Amplitude ring params (independent)
    amp_m_arcs_per_class = 6
    amp_gap_frac = 0.15

    # Amplitude ring controls
    amp_ring_radius = 1  # smaller = subtler deformation
    amp_scale_a1 = 0.2
    amp_scale_a2 = 0.3

    difficulty_sharp = 0.2  # try 0.3–0.8; 0 = no sharpening difference

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
    )
    plot_and_record_video(num_unsup_epochs=20,
                          num_sup_epochs=10,
                          num_batches=7,
                          out_video="../plots/training_evolution_conv.mp4",
                          dataset=ds)
