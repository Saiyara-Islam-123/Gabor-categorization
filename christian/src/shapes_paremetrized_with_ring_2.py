"""
shape_deform_dataset.py
-----------------------

Two-class *image* dataset built from an isotropic 2-mode shape generator whose
latent parameters lie on alternating arcs of the unit circle or 4D sphere.
Supports independent or fixed harmonic phases, multiple amplitude modes,
and joint phase–amplitude parity coupling.
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.path import Path

# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance_matrix(embeddings1: torch.Tensor,
                              embeddings2: torch.Tensor = None,
                              normalize: bool = True,
                              eps: float = 1e-8):
    """
    Compute pairwise squared Euclidean distances with optional L2 normalization.
    """
    if normalize:
        embeddings1 = F.normalize(embeddings1, p=2, dim=1, eps=eps)
        if embeddings2 is not None:
            embeddings2 = F.normalize(embeddings2, p=2, dim=1, eps=eps)
    if embeddings2 is None:
        sq_norms = (embeddings1 ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norms + sq_norms.T - 2 * embeddings1 @ embeddings1.T
    else:
        sq_norms1 = (embeddings1 ** 2).sum(dim=1, keepdim=True)
        sq_norms2 = (embeddings2 ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norms1 + sq_norms2.T - 2 * embeddings1 @ embeddings2.T
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return dist_sq


# ─────────────────────────────────────────────────────────────────────────────
# Arc utilities
# ─────────────────────────────────────────────────────────────────────────────

def _alternating_arcs_boundaries(m_arcs_per_class: int,
                                 phase_rad: float = 0.0,
                                 gap_frac: float = 0.0):
    assert m_arcs_per_class >= 1
    total_arcs = 2 * m_arcs_per_class
    base_width = 2 * math.pi / total_arcs
    gap = gap_frac * base_width
    arcs0, arcs1 = [], []
    for k in range(total_arcs):
        start = phase_rad + k * base_width + gap / 2
        end = phase_rad + (k + 1) * base_width - gap / 2
        if k % 2 == 0:
            arcs0.append((start % (2*math.pi), end % (2*math.pi)))
        else:
            arcs1.append((start % (2*math.pi), end % (2*math.pi)))
    return arcs0, arcs1, base_width

def _sample_uniform_angle_on_arc(rng, start, end):
    if end < start:
        end += 2 * math.pi
    theta = rng.uniform(start, end)
    return theta % (2 * math.pi)


def _angle_in_arc(theta: float, start: float, end: float) -> bool:
    """Check if theta ∈ [start,end] with wrap; all angles in [0, 2π)."""
    if end >= start:
        return (theta >= start) and (theta <= end)
    else:
        return (theta >= start) or (theta <= end)

def _merge_alternating(arcs0: list, arcs1: list) -> list:
    """Interleave arcs0/arcs1 back into the original ordered list of 2m arcs."""
    arcs = []
    K0, K1 = len(arcs0), len(arcs1)
    for k in range(K0 + K1):
        if k % 2 == 0:
            arcs.append(arcs0[k // 2])
        else:
            arcs.append(arcs1[k // 2])
    return arcs



# ─────────────────────────────────────────────────────────────────────────────
# Shape rendering
# ─────────────────────────────────────────────────────────────────────────────

def _shape_boundary(R, k1, k2, num_samples=512, rot_rad=0.0, mode="phase",
                    x=None, y=None, a1=0.0, a2=0.0,
                    alpha_x=0.0, beta_y=0.0, phi1=None, phi2=None,sharp_s: float = 0.0, sharpen_with_abs: bool = True):
    theta = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    theta_rot = theta + rot_rad
    r = np.ones_like(theta_rot)
    if mode == "phase":
        r += a1*(x*np.cos(k1*theta_rot)+y*np.sin(k1*theta_rot))
        r += a2*(x*np.sin(k2*theta_rot)-y*np.cos(k2*theta_rot))
    elif mode == "mode":
        r += alpha_x*x*np.cos(k1*theta_rot)
        r += beta_y*y*np.cos(k2*theta_rot)
    elif mode == "indep_phase":
        r += a1*np.cos(k1*(theta_rot-phi1))
        r += a2*np.cos(k2*(theta_rot-phi2))
        # --- NEW: class-controlled sharpening via 2× harmonics ---
        if sharp_s != 0.0:
            a1_w = abs(a1) if sharpen_with_abs else a1
            a2_w = abs(a2) if sharpen_with_abs else a2
            r += sharp_s * (a1_w * np.cos(2 * k1 * (theta_rot - phi1)) +
                            a2_w * np.cos(2 * k2 * (theta_rot - phi2)))
    else:
        raise ValueError("Invalid mode")
    xs, ys = R*r*np.cos(theta_rot), R*r*np.sin(theta_rot)
    return xs, ys

def _rasterize_filled_polygon(H, W, xs, ys, cx, cy, scale_px,
                              fill_value=1.0, bg=0.0):
    verts = np.stack([cx + scale_px*xs, cy + scale_px*ys], axis=1)
    poly = Path(verts, closed=True)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([xx + 0.5, yy + 0.5], axis=-1).reshape(-1, 2)
    mask = poly.contains_points(pts).reshape(H, W)
    img = np.full((H, W), bg, np.float32)
    img[mask] = fill_value
    return img

def _render_shape_image(H, W, ring_radius_norm, k1, k2, mode, x=None, y=None,
                        a1=0.0, a2=0.0, alpha_x=0.0, beta_y=0.0,
                        phi1=None, phi2=None,sharp_s=0.0,
                        sharpen_with_abs=True, intensity=1.0, bg=0.0,
                        norm="max", global_rot_rad=0.0):

    xs, ys = _shape_boundary(R=1.0, k1=k1, k2=k2, num_samples=512,
                             rot_rad=global_rot_rad, mode=mode, x=x, y=y,
                             a1=a1, a2=a2, alpha_x=alpha_x, beta_y=beta_y,
                             phi1=phi1, phi2=phi2,sharp_s=sharp_s, sharpen_with_abs=sharpen_with_abs)
    cx, cy = (W-1)/2, (H-1)/2
    img = _rasterize_filled_polygon(H, W, xs, ys, cx, cy, ring_radius_norm,
                                    fill_value=intensity, bg=bg)
    if norm == "max":
        m = img.max()
        if m > 0: img /= m
    elif norm == "l2":
        n = np.linalg.norm(img)
        if n > 0: img /= n
    return np.clip(img, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class ShapeDeformDataset(Dataset):
    def __init__(self, nA, nB, m_arcs_per_class=2, gap_frac=0.2, phase_deg=0.0,
                 image_size=128, ring_radius_px=40,
                 k_lobes=3, k2_lobes=5,
                 a1=0.1, a2=0.1,
                 mode="indep_phase",
                 alpha_x=0.1, beta_y=0.1,
                 intensity=1.0, bg=0.0, seed=42,
                 norm="max", global_rot="none",
                 # new amplitude controls
                 amp_mode="fixed", amp_class_coupling="none",
                 # NEW: amplitude ring params (independent of phase ring)
                 difficulty_sharp=0.0,
                 amp_ring_radius=0.15, amp_scale_a1=1.0, amp_scale_a2=1.0,
                 amp_m_arcs_per_class=None, amp_gap_frac=None,
                 # new phase controls
                 phase_mode="independent", phase_fixed_values=(0.0, 0.0)):

        # independent amplitude arc controls (fallback to phase values)
        if amp_m_arcs_per_class is None:
            amp_m_arcs_per_class = m_arcs_per_class
        if amp_gap_frac is None:
            amp_gap_frac = gap_frac
        self.amp_m_arcs_per_class = int(amp_m_arcs_per_class)
        self.amp_gap_frac = float(amp_gap_frac)


        rng = np.random.default_rng(seed)
        self.H = self.W = image_size
        self.ring_radius_px = ring_radius_px
        self.k1, self.k2 = k_lobes, k2_lobes
        self.a1, self.a2 = a1, a2
        self.mode = mode
        self.alpha_x, self.beta_y = alpha_x, beta_y
        self.intensity, self.bg, self.norm = intensity, bg, norm
        self.global_rot = global_rot
        self.amp_mode = amp_mode
        self.amp_ring_radius = amp_ring_radius
        self.amp_scale_a1 = amp_scale_a1
        self.amp_scale_a2 = amp_scale_a2
        self.difficulty_sharp = difficulty_sharp

        self.amp_class_coupling = amp_class_coupling
        self.phase_mode = phase_mode
        self.phase_fixed_values = phase_fixed_values


        phase_rad = math.radians(phase_deg)
        arcs_A, arcs_B, width = _alternating_arcs_boundaries(
            m_arcs_per_class, phase_rad, gap_frac)

        # --- amplitude arcs (shared for both classes so parity can work)
        amp_arcs_A, amp_arcs_B, _ = _alternating_arcs_boundaries(
            self.amp_m_arcs_per_class, 0.0, self.amp_gap_frac
        )
        amp_arcs_all = _merge_alternating(amp_arcs_A, amp_arcs_B)  # ordered 2m_amp arcs


        def _sample_amp_ring_shared(n, arcs_all, rng):
            j = rng.integers(0, len(arcs_all), size=n)
            chi = np.empty(n, dtype=np.float32)
            for i in range(n):
                s, e = arcs_all[j[i]]
                chi[i] = _sample_uniform_angle_on_arc(rng, s, e)
            # amplitudes live on a 2D ring here (use cos/sin as a convenient param)
            amps = np.stack([np.cos(chi), np.sin(chi)], axis=1).astype(np.float32)
            return amps, j.astype(np.int32), chi

        if amp_mode == "fixed":
            A_amp = np.full((nA, 2), [a1, a2], np.float32)
            B_amp = np.full((nB, 2), [a1, a2], np.float32)
            # derive χ / j_amp for labeling/plots even if fixed
            A_amp_idx = np.zeros(nA, dtype=np.int32)
            B_amp_idx = np.zeros(nB, dtype=np.int32)
            A_amp_ang = np.zeros(nA, dtype=np.float32)
            B_amp_ang = np.zeros(nB, dtype=np.float32)

        elif amp_mode == "independent_uniform":
            A_amp = rng.uniform(-abs(a1), abs(a1), (nA, 2)).astype(np.float32)
            B_amp = rng.uniform(-abs(a1), abs(a1), (nB, 2)).astype(np.float32)
            A_amp_ang = np.arctan2(A_amp[:, 1], A_amp[:, 0]).astype(np.float32) % (2 * np.pi)
            B_amp_ang = np.arctan2(B_amp[:, 1], B_amp[:, 0]).astype(np.float32) % (2 * np.pi)
            # quantize χ into the same 2*m_amp amplitude arcs
            total_amp_arcs = 2 * self.amp_m_arcs_per_class
            arc_w = 2 * np.pi / total_amp_arcs
            A_amp_idx = np.floor(A_amp_ang / arc_w).astype(np.int32) % total_amp_arcs
            B_amp_idx = np.floor(B_amp_ang / arc_w).astype(np.int32) % total_amp_arcs

        elif amp_mode == "ring_shared_arcs":
            A_amp_raw, A_amp_idx, A_amp_ang = _sample_amp_ring_shared(nA, amp_arcs_all, rng)
            B_amp_raw, B_amp_idx, B_amp_ang = _sample_amp_ring_shared(nB, amp_arcs_all, rng)

            # Apply amplitude ring scaling (keep sign)
            def _scale_ring(amps):
                return np.stack([
                    self.amp_ring_radius * self.amp_scale_a1 * amps[:, 0],
                    self.amp_ring_radius * self.amp_scale_a2 * amps[:, 1]
                ], axis=1).astype(np.float32)

            A_amp = _scale_ring(A_amp_raw)
            B_amp = _scale_ring(B_amp_raw)

        else:
            raise ValueError("invalid amp_mode")

        # --- phase arcs (A/B lists + merged ordered list)
        arcs_A, arcs_B, width = _alternating_arcs_boundaries(m_arcs_per_class, phase_rad, gap_frac)
        phase_arcs_all = _merge_alternating(arcs_A, arcs_B)  # ordered 2m arcs

        def _sample_phase_with_index(n, arcs_A, arcs_B, arcs_all, rng):
            # Sample thetas exactly as before (A from arcs_A, B from arcs_B),
            # but return their global arc index i_phase in 0..(2m-1).
            thetas_A = np.empty(n, dtype=np.float32)
            i_phase_A = np.empty(n, dtype=np.int32)
            for t in range(n):
                i_local = rng.integers(len(arcs_A))
                s, e = arcs_A[i_local]
                th = _sample_uniform_angle_on_arc(rng, s, e)
                thetas_A[t] = th
                i_phase_A[t] = 2 * i_local  # even slots in merged list

            thetas_B = np.empty(n, dtype=np.float32)
            i_phase_B = np.empty(n, dtype=np.int32)
            for t in range(n):
                i_local = rng.integers(len(arcs_B))
                s, e = arcs_B[i_local]
                th = _sample_uniform_angle_on_arc(rng, s, e)
                thetas_B[t] = th
                i_phase_B[t] = 2 * i_local + 1  # odd slots in merged list
            return thetas_A, i_phase_A, thetas_B, i_phase_B

        # --- sample phases
        if phase_mode == "fixed":
            phi1_A = np.full(nA, phase_fixed_values[0], dtype=np.float32)
            phi2_A = np.full(nA, phase_fixed_values[1], dtype=np.float32)
            phi1_B = np.full(nB, phase_fixed_values[0], dtype=np.float32)
            phi2_B = np.full(nB, phase_fixed_values[1], dtype=np.float32)
            i_phase_A = np.zeros(nA, dtype=np.int32)
            i_phase_B = np.zeros(nB, dtype=np.int32)
        else:
            # as before…
            phi1_A, i_phase_A1, phi1_B, i_phase_B1 = _sample_phase_with_index(nA, arcs_A, arcs_B, phase_arcs_all, rng)
            phi2_A, i_phase_A2, phi2_B, i_phase_B2 = _sample_phase_with_index(nA, arcs_A, arcs_B, phase_arcs_all, rng)

            i_phase_A, i_phase_B = i_phase_A1, i_phase_B1

        # ─────────────────────────────────────────────────────────────────────────────
        # One phase ring, one amplitude ring, and ONE active ring for plotting/back-compat
        # ─────────────────────────────────────────────────────────────────────────────

        # 1) Phase ring (φ1) is always available (project S³ → unit circle via φ1)
        self.phase_ring_A = np.stack([np.cos(phi1_A), np.sin(phi1_A)], axis=1).astype(np.float32)
        self.phase_ring_B = np.stack([np.cos(phi1_B), np.sin(phi1_B)], axis=1).astype(np.float32)

        # 2) Amplitude ring (χ) works for all amp_mode:
        #    - ring_shared_arcs: use sampled A_amp_ang/B_amp_ang
        #    - fixed / independent_uniform: derive χ from raw [a1, a2] by atan2(a2, a1)
        if 'A_amp_ang' in locals() and 'B_amp_ang' in locals() and A_amp_ang is not None and B_amp_ang is not None:
            chi_A = A_amp_ang
            chi_B = B_amp_ang
        else:
            # Derive from raw amplitudes (handles fixed / independent_uniform)
            chi_A = np.arctan2(A_amp[:, 1], A_amp[:, 0])
            chi_B = np.arctan2(B_amp[:, 1], B_amp[:, 0])

        chi_A = (chi_A + 2 * np.pi) % (2 * np.pi)
        chi_B = (chi_B + 2 * np.pi) % (2 * np.pi)

        self.amp_ring_A = np.stack([np.cos(chi_A), np.sin(chi_A)], axis=1).astype(np.float32)
        self.amp_ring_B = np.stack([np.cos(chi_B), np.sin(chi_B)], axis=1).astype(np.float32)

        # 3) Choose the ACTIVE ring for external plotting APIs:
        #    - phase_mode == "independent" → phase ring is active
        #    - phase_mode == "fixed"       → amplitude ring is active
        if getattr(self, "phase_mode", "independent") == "fixed":
            self.latents_A, self.latents_B = self.amp_ring_A, self.amp_ring_B
            self.ring_kind = "amplitude"
        else:
            self.latents_A, self.latents_B = self.phase_ring_A, self.phase_ring_B
            self.ring_kind = "phase"

        # --- class labels
            # phase is constant → classes from amplitude-ring parity only
        if phase_mode == "fixed":
            # phase is constant → classes from amplitude-ring parity ONLY
            A_lbl = (A_amp_idx % 2).astype(np.int32)
            B_lbl = (B_amp_idx % 2).astype(np.int32)
        else:
            if amp_class_coupling == "parity":
                A_lbl = ((i_phase_A + A_amp_idx) & 1).astype(np.int32)
                B_lbl = ((i_phase_B + B_amp_idx) & 1).astype(np.int32)
            else:
                A_lbl = (i_phase_A & 1).astype(np.int32)
                B_lbl = (i_phase_B & 1).astype(np.int32)

        # store latents
        self.amp_latents = np.concatenate([A_amp, B_amp], 0)
        self.amp_angles = np.concatenate([A_amp_ang, B_amp_ang], 0)
        self.latents = np.stack([
            np.cos(np.concatenate([phi1_A, phi1_B])),
            np.sin(np.concatenate([phi1_A, phi1_B])),
            np.cos(np.concatenate([phi2_A, phi2_B])),
            np.sin(np.concatenate([phi2_A, phi2_B]))
        ], 1)

        # --- Unified ring latents for plotting/consumers ---
        if self.phase_mode == "fixed":
            if 'A_amp_ang' in locals() and 'B_amp_ang' in locals():
                chi_A = (A_amp_ang + 2 * np.pi) % (2 * np.pi)
                chi_B = (B_amp_ang + 2 * np.pi) % (2 * np.pi)
            else:
                chi_A = (np.arctan2(A_amp[:, 1], A_amp[:, 0]) + 2 * np.pi) % (2 * np.pi)
                chi_B = (np.arctan2(B_amp[:, 1], B_amp[:, 0]) + 2 * np.pi) % (2 * np.pi)

            self.latents_A = np.stack([np.cos(chi_A), np.sin(chi_A)], axis=1).astype(np.float32)
            self.latents_B = np.stack([np.cos(chi_B), np.sin(chi_B)], axis=1).astype(np.float32)
        else:
            self.latents_A = np.stack([np.cos(phi1_A), np.sin(phi1_A)], axis=1).astype(np.float32)
            self.latents_B = np.stack([np.cos(phi1_B), np.sin(phi1_B)], axis=1).astype(np.float32)

        labels = np.concatenate([A_lbl, B_lbl], 0)
        imgs = []
        for i in range(nA + nB):
            ph1 = phi1_A[i] if i < nA else phi1_B[i - nA]
            ph2 = phi2_A[i] if i < nA else phi2_B[i - nA]
            aa1, aa2 = self.amp_latents[i]

            lab = labels[i]  # 0 or 1
            s_sharp = 0.0 if lab == 0 else self.difficulty_sharp  # Class 2 sharper

            img = _render_shape_image(self.H, self.W,
                                      self.ring_radius_px, self.k1, self.k2,
                                      mode="indep_phase",
                                      a1=aa1, a2=aa2,
                                      phi1=ph1, phi2=ph2,
                                      sharp_s=s_sharp,
                                      sharpen_with_abs=True,
                                      intensity=1.0, bg=0.0,
                                      norm=self.norm, global_rot_rad=0.0,
                                      )
            imgs.append(img)
        self.images = np.stack(imgs)
        self.labels = labels.astype(np.int64)

        # store amplitude diagnostics for plotting
        self.amp_latents = np.concatenate([A_amp, B_amp], 0)  # [N,2]
        self.amp_angles = np.concatenate([A_amp_ang, B_amp_ang], 0)  # [N]
        self.phase_mode = phase_mode  # for plots
        self.m_arcs_per_class = m_arcs_per_class  # phase arcs (for reference)
        self.amp_m_arcs_per_class = self.amp_m_arcs_per_class
        self.amp_gap_frac = self.amp_gap_frac

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return torch.from_numpy(self.images[i])[None], torch.tensor(self.labels[i])

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics: squared distances (latent, amplitude-latent, pixel)
# ─────────────────────────────────────────────────────────────────────────────

def print_latent_distance_stats(dataset: "ShapeDeformDataset", normalize=True):
    """
    Mean within/between *squared* Euclidean distances in (phase) latent space.
    Uses dataset.latents = [cos φ1, sin φ1, cos φ2, sin φ2].
    If normalize=True, it's computed on the unit sphere (equals 2 - 2·cos).
    """
    Z = dataset.latents.astype(np.float32)        # [N, 4] in indep_phase
    labs = dataset.labels
    A = Z[labs == 0]
    B = Z[labs == 1]
    A_t = torch.from_numpy(A.copy())
    B_t = torch.from_numpy(B.copy())

    D_within_A = euclidean_distance_matrix(A_t, normalize=normalize)
    D_within_B = euclidean_distance_matrix(B_t, normalize=normalize)
    iu_A = torch.triu_indices(D_within_A.size(0), D_within_A.size(1), offset=1)
    iu_B = torch.triu_indices(D_within_B.size(0), D_within_B.size(1), offset=1)
    mean_within_A = D_within_A[iu_A[0], iu_A[1]].mean().item() if iu_A.numel() else 0.0
    mean_within_B = D_within_B[iu_B[0], iu_B[1]].mean().item() if iu_B.numel() else 0.0
    D_between = euclidean_distance_matrix(A_t, B_t, normalize=normalize).mean().item()
    overall_within = 0.5 * (mean_within_A + mean_within_B)

    tag = "unit-sphere" if normalize else "raw"
    print(f"\n=== Latent-space *SQUARED* Euclidean distances ({tag}) ===")
    print(f"Within Class 0: {mean_within_A:.4f}")
    print(f"Within Class 1: {mean_within_B:.4f}")
    print(f"Overall within: {overall_within:.4f}")
    print(f"Between (0 vs 1): {D_between:.4f}")


def print_amplitude_distance_stats(dataset: "ShapeDeformDataset", normalize=False):
    """
    Mean within/between *squared* Euclidean distances in amplitude latent space.
    - If amp_mode='ring_shared_arcs', we have a ring angle χ; we use 2D ring coords [cos χ, sin χ].
    - Else (fixed / independent_uniform), we use raw [a1, a2] magnitudes.
    """
    if hasattr(dataset, "amp_angles") and np.any(dataset.amp_angles != 0):
        chi = dataset.amp_angles.astype(np.float32)
        Z = np.stack([np.cos(chi), np.sin(chi)], axis=1).astype(np.float32)
    else:
        # raw amplitude magnitudes (sign ignored in Part 1 construction)
        Z = dataset.amp_latents.astype(np.float32)

    labs = dataset.labels
    A = Z[labs == 0]
    B = Z[labs == 1]
    A_t = torch.from_numpy(A.copy())
    B_t = torch.from_numpy(B.copy())

    D_within_A = euclidean_distance_matrix(A_t, normalize=normalize)
    D_within_B = euclidean_distance_matrix(B_t, normalize=normalize)
    iu_A = torch.triu_indices(D_within_A.size(0), D_within_A.size(1), offset=1)
    iu_B = torch.triu_indices(D_within_B.size(0), D_within_B.size(1), offset=1)
    mean_within_A = D_within_A[iu_A[0], iu_A[1]].mean().item() if iu_A.numel() else 0.0
    mean_within_B = D_within_B[iu_B[0], iu_B[1]].mean().item() if iu_B.numel() else 0.0
    D_between = euclidean_distance_matrix(A_t, B_t, normalize=normalize).mean().item()
    overall_within = 0.5 * (mean_within_A + mean_within_B)

    tag = "unit-sphere" if normalize else "raw"
    print(f"\n=== Amplitude-space *SQUARED* Euclidean distances ({tag}) ===")
    print(f"Within Class 0: {mean_within_A:.4f}")
    print(f"Within Class 1: {mean_within_B:.4f}")
    print(f"Overall within: {overall_within:.4f}")
    print(f"Between (0 vs 1): {D_between:.4f}")


def print_image_distance_stats(dataset: "ShapeDeformDataset"):
    """
    Mean within/between *squared* Euclidean distances in pixel space.
    Computed on L2-normalized grayscale vectors (2 - 2·cosine).
    """
    imgs = dataset.images.astype(np.float32)  # [N,H,W]
    A = imgs[dataset.labels == 0].reshape(-1, dataset.H * dataset.W)
    B = imgs[dataset.labels == 1].reshape(-1, dataset.H * dataset.W)
    A_t = torch.from_numpy(A.copy())
    B_t = torch.from_numpy(B.copy())

    D_within_A = euclidean_distance_matrix(A_t, normalize=True)
    D_within_B = euclidean_distance_matrix(B_t, normalize=True)
    iu_A = torch.triu_indices(D_within_A.size(0), D_within_A.size(1), offset=1)
    iu_B = torch.triu_indices(D_within_B.size(0), D_within_B.size(1), offset=1)
    mean_within_A = D_within_A[iu_A[0], iu_A[1]].mean().item() if iu_A.numel() else 0.0
    mean_within_B = D_within_B[iu_B[0], iu_B[1]].mean().item() if iu_B.numel() else 0.0
    D_between = euclidean_distance_matrix(A_t, B_t, normalize=True).mean().item()
    overall_within = 0.5 * (mean_within_A + mean_within_B)

    print("\n=== Pixel-space *SQUARED* Euclidean distances (L2-normalized) ===")
    print(f"Within Class 0: {mean_within_A:.4f}")
    print(f"Within Class 1: {mean_within_B:.4f}")
    print(f"Overall within: {overall_within:.4f}")
    print(f"Between (0 vs 1): {D_between:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def show_examples(dataset: "ShapeDeformDataset", n=8):
    """
    Top row: n exemplars from Class 0.
    Bottom row: for each exemplar, the closest example from Class 1
                in pixel space (squared Euclidean on L2-normalized vectors).
    """
    A_all = np.where(dataset.labels == 0)[0]
    B_all = np.where(dataset.labels == 1)[0]
    if len(A_all) == 0 or len(B_all) == 0:
        raise ValueError("Both classes must be present.")
    n = min(n, len(A_all), len(B_all))
    A_idx = A_all[:n]

    H, W = dataset.images.shape[1], dataset.images.shape[2]
    A_vecs = dataset.images[A_idx].reshape(n, -1).astype(np.float32)
    B_vecs = dataset.images[B_all].reshape(len(B_all), -1).astype(np.float32)
    def _l2norm(X, eps=1e-12):
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm = np.maximum(nrm, eps)
        return X / nrm
    A_n = _l2norm(A_vecs)
    B_n = _l2norm(B_vecs)
    cos = A_n @ B_n.T
    D2  = 2.0 - 2.0 * np.clip(cos, -1.0, 1.0)
    nn_in_B = D2.argmin(axis=1)
    nn_D2   = D2[np.arange(n), nn_in_B]

    fig, axes = plt.subplots(2, n, figsize=(1.5*n, 3.4), constrained_layout=True)

    def tint(gray, color):
        r = gray * color[0]; g = gray * color[1]; b = gray * color[2]
        return np.stack([r, g, b], axis=-1)

    green = (0.0, 1.0, 0.0)
    lime  = (0.85, 1.0, 0.0)

    for j, idx in enumerate(A_idx):
        rgb = tint(dataset.images[idx], green)
        ax = axes[0, j]; ax.imshow(rgb, vmin=0, vmax=1); ax.axis("off")
        ax.set_title("C0 exemplar", color="green", fontsize=9)

    for j in range(n):
        b_idx = B_all[nn_in_B[j]]; d2 = nn_D2[j]
        rgb = tint(dataset.images[b_idx], lime)
        ax = axes[1, j]; ax.imshow(rgb, vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"C1 nearest\nD²={d2:.3f}", color="#7a8c00", fontsize=9)

    plt.show()


def _phase_angles_from_latents(dataset: "ShapeDeformDataset"):
    """Recover φ1 and φ2 from stored 4D latents."""
    Z = dataset.latents
    phi1 = np.arctan2(Z[:,1], Z[:,0])  # atan2(sinφ1, cosφ1)
    phi2 = np.arctan2(Z[:,3], Z[:,2])
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    phi2 = (phi2 + 2*np.pi) % (2*np.pi)
    return phi1, phi2


def plot_latent_scatter(dataset, title="Latent ring by class"):
    """
    Unified ring plot:
      - If phase_mode='independent' → shows the phase ring (φ₁)
      - If phase_mode='fixed'       → shows the amplitude ring (χ)
    Always colors by dataset.labels (green=0, lime=1).
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    labs = _np.asarray(dataset.labels)

    # --- pick a label for the plot title ---
    if getattr(dataset, "phase_mode", "independent") == "fixed":
        ring_label = "Amplitude ring (χ)"
    else:
        ring_label = "Phase ring (φ₁)"

    fig, ax = _plt.subplots(figsize=(6, 6))
    t = _np.linspace(0, 2*_np.pi, 400)
    ax.plot(_np.cos(t), _np.sin(t), lw=1.0, color="black", alpha=0.6)

    # unified latents from dataset
    A, B = dataset.latents_A, dataset.latents_B
    ax.scatter(A[:, 0], A[:, 1], s=10, color="green", alpha=0.7, label="Class 0")
    ax.scatter(B[:, 0], B[:, 1], s=10, color="lime",  alpha=0.7, label="Class 1")

    ax.set_aspect("equal", "box")
    ax.set_title(f"{title} — {ring_label}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper right")
    _plt.show()

def plot_phase_amp_torus(dataset: "ShapeDeformDataset", alpha=0.6):
    """
    Torus-strip scatter: x = phase angle (φ1), y = amplitude angle (χ).
    Works for any amp_mode; if no ring angles saved, χ is derived from [a1,a2] via atan2.
    """
    labs = dataset.labels
    phi1, _ = _phase_angles_from_latents(dataset)

    if hasattr(dataset, "amp_angles") and np.any(dataset.amp_angles != 0):
        chi = dataset.amp_angles
    else:
        # derive amplitude angle from raw amplitudes
        a = dataset.amp_latents
        chi = np.arctan2(a[:,1], a[:,0])
    chi = (chi + 2*np.pi) % (2*np.pi)
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(phi1[labs==0], chi[labs==0], s=8, alpha=alpha, label="Class 0", color="green")
    ax.scatter(phi1[labs==1], chi[labs==1], s=8, alpha=alpha, label="Class 1", color="lime")
    ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)
    ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    ax.set_xticks(ticks); ax.set_xticklabels(["0","π/2","π","3π/2","2π"])
    ax.set_yticks(ticks); ax.set_yticklabels(["0","π/2","π","3π/2","2π"])
    ax.set_xlabel("Phase angle φ1"); ax.set_ylabel("Amplitude angle χ")
    ax.set_title("Phase–Amplitude Torus (φ1 vs χ)")
    ax.grid(True, ls="--", alpha=0.3); ax.legend()
    plt.show()


def plot_phase_ring_with_amp_bands(dataset: "ShapeDeformDataset", band_gap=0.15, alpha=0.8):
    """
    Concentric ring: angle = phase angle φ1; radius band encodes amplitude parity.
    If amplitude ring angles are unavailable, parity is derived from atan2(a1,a2)
    and split into 2*m arcs using the dataset's m_arcs_per_class as m.
    """
    labs = dataset.labels
    phi1, _ = _phase_angles_from_latents(dataset)
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    X = np.stack([np.cos(phi1), np.sin(phi1)], axis=1)

    # derive amplitude angle
    if hasattr(dataset, "amp_angles") and np.any(dataset.amp_angles != 0):
        chi = (dataset.amp_angles + 2*np.pi) % (2*np.pi)
    else:
        a = dataset.amp_latents
        chi = (np.arctan2(a[:,1], a[:,0]) + 2*np.pi) % (2*np.pi)

    # Quantize χ into 2*m_amp arcs and use parity for banding
    m_amp = getattr(dataset, "amp_m_arcs_per_class", 2)
    total_arcs = 2 * max(1, int(m_amp))
    arc_w = 2*np.pi / total_arcs
    j = np.floor(chi / arc_w).astype(int) % total_arcs
    even = (j % 2 == 0)


    r_even = 1.0
    r_odd  = max(0.0, 1.0 - band_gap)
    XY = X.copy()
    XY[ even] *= r_even
    XY[~even] *= r_odd

    fig, ax = plt.subplots(figsize=(6, 6))
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), lw=1.0, color="black", alpha=0.6)

    ax.scatter(XY[(labs==0) &  even,0], XY[(labs==0) &  even,1], s=12, alpha=alpha, color="green",    label="C0 (even χ-arc)")
    ax.scatter(XY[(labs==0) & (~even),0], XY[(labs==0) & (~even),1], s=12, alpha=alpha, color="green",    marker="x", label="C0 (odd χ-arc)")
    ax.scatter(XY[(labs==1) &  even,0], XY[(labs==1) &  even,1], s=12, alpha=alpha, color="lime", label="C1 (even χ-arc)")
    ax.scatter(XY[(labs==1) & (~even),0], XY[(labs==1) & (~even),1], s=12, alpha=alpha, color="lime", marker="x", label="C1 (odd χ-arc)")

    ax.set_aspect("equal", "box")
    ax.set_title("Phase Ring with Amplitude Parity Bands (via χ)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers + main
# ─────────────────────────────────────────────────────────────────────────────

def load_shape_deform_data(
    nA=1000, nB=1000,
    m_arcs_per_class=2, gap_frac=0.2, phase_deg=0.0,
    image_size=128, ring_radius_px=40,
    k1=3, k2=5,
    a1=0.1, a2=0.1,
    mode="indep_phase",
    alpha_x=0.1, beta_y=0.1,
    intensity=1.0, bg=0.0, norm="max", global_rot="none",
    amp_mode="fixed", amp_class_coupling="none",
    difficulty_sharp=0.0,
    amp_ring_radius=0.15,
    amp_scale_a1=1, amp_scale_a2=1,
    # NEW: independent amplitude ring controls
    amp_m_arcs_per_class=None, amp_gap_frac=None,
    phase_mode="independent", phase_fixed_values=(0.0, 0.0),
    batch_size=256, seed=None
):

    if seed is None:
        seed = np.random.randint(1_000_000)
    ds = ShapeDeformDataset(
        nA=nA, nB=nB,
        m_arcs_per_class=m_arcs_per_class, gap_frac=gap_frac, phase_deg=phase_deg,
        image_size=image_size, ring_radius_px=ring_radius_px,
        k_lobes=k1, k2_lobes=k2,
        a1=a1, a2=a2,
        mode=mode, alpha_x=alpha_x, beta_y=beta_y,
        intensity=intensity, bg=bg, seed=seed, norm=norm, global_rot=global_rot,
        amp_mode=amp_mode, amp_class_coupling=amp_class_coupling,
        difficulty_sharp=difficulty_sharp,
        amp_ring_radius=amp_ring_radius,
        amp_scale_a1=amp_scale_a1, amp_scale_a2=amp_scale_a2,
        amp_m_arcs_per_class=amp_m_arcs_per_class, amp_gap_frac=amp_gap_frac,  # NEW
        phase_mode=phase_mode, phase_fixed_values=phase_fixed_values)
    torch.manual_seed(seed)
    N = len(ds)
    n_train = int(0.8 * N); n_val = int(0.1 * N); n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test])
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    testloader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader, ds


def main():
    # ---- Configurable parameters ----
    nA = 500
    nB = nA
    m_arcs_per_class = 2
    gap_frac = 0.5
    phase_deg = 0.0

    # Amplitude ring params (independent)
    amp_m_arcs_per_class = 6
    amp_gap_frac = 0.15

    # Amplitude ring controls
    amp_ring_radius = 0.8  # smaller = subtler deformation
    amp_scale_a1 = 0.2
    amp_scale_a2 = 0.6

    difficulty_sharp = 0.15  # try 0.3–0.8; 0 = no sharpening difference

    image_size = 128
    ring_radius_px = 40

    k1, k2 = 3, 5
    a1, a2 = 0.1, 0.2

    # Phase config
    phase_mode = "independent"                  # "independent" | "fixed"
    phase_fixed_values = (0, 0)   # used iff phase_mode == "fixed"

    # Amplitudes config
    amp_mode = "ring_shared_arcs"         # "fixed" | "independent_uniform" | "ring_shared_arcs"
    amp_class_coupling = "parity"         # "none" | "parity"

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

    print(f"Dataset size: {len(ds)} images  |  Image shape: {ds.images.shape[1:]}")
    print_latent_distance_stats(ds, normalize=True)   # phase latent
    print_amplitude_distance_stats(ds, normalize=False)  # amplitude latent
    print_image_distance_stats(ds)                    # pixel space

    show_examples(ds, n=12)
    plot_latent_scatter(ds, title="Latent projection by class")
    plot_phase_amp_torus(ds)
    plot_phase_ring_with_amp_bands(ds, band_gap=0.15)


if __name__ == "__main__":
    main()
