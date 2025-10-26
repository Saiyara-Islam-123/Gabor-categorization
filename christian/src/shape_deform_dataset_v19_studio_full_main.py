#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shape_deform_dataset_v19_studio_full_main.py
--------------------------------------------
Version 19 — Studio generates EVERYTHING; dataset just collects & packages.
No argparse. Hard-coded main that exposes **all** Studio sliders/knobs,
including tinys and both bases' controls.

- Uses Studio.resample_exemplars() repeatedly to gather any number of specs.
- Uses Studio._render_with_spec(...) to render each exemplar (no dataset-side logic).
- Keeps: stats printouts, example+nearest plotting, Excel+PNG export.
- Main includes every Studio control like your older mains.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import os, math, numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path as _Path

# Headless Qt (no visible UI needed)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6 import QtWidgets

# Import Studio (single source of truth)
import Studio_fix_opaque_blackedge_None_smallk_fill_freqsliders_fast as StudioMod

# Single app + single Studio instance
_QTAPP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
_STUDIO = StudioMod.ShapeStudio(); _STUDIO.setVisible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance_matrix(embeddings1: torch.Tensor,
                              embeddings2: torch.Tensor = None,
                              normalize: bool = True,
                              eps: float = 1e-8):
    if normalize:
        embeddings1 = F.normalize(embeddings1, p=2, dim=1, eps=eps)
        if embeddings2 is not None:
            embeddings2 = F.normalize(embeddings2, p=2, dim=1, eps=eps)
    if embeddings2 is None:
        sq = (embeddings1**2).sum(1, keepdim=True)
        D = sq + sq.T - 2*embeddings1 @ embeddings1.T
    else:
        sq1 = (embeddings1**2).sum(1, keepdim=True)
        sq2 = (embeddings2**2).sum(1, keepdim=True)
        D = sq1 + sq2.T - 2*embeddings1 @ embeddings2.T
    return torch.clamp(D, min=0.0)


def _rasterize_filled_polygon(H, W, xs, ys, cx, cy, scale_px, fill_value=1.0, bg=0.0):
    from matplotlib.path import Path as _PathMP
    verts = np.stack([cx + scale_px*xs, cy + scale_px*ys], axis=1)
    poly = _PathMP(verts, closed=True)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([xx + 0.5, yy + 0.5], axis=-1).reshape(-1, 2)
    mask = poly.contains_points(pts).reshape(H, W)
    img = np.full((H, W), bg, np.float32); img[mask] = fill_value
    return img


class _SpecApply:
    """Temporarily apply a Studio spec dict: {'phi':(idx,t), 'amp':(idx,t), 'freq':(k1,k2)}"""
    def __init__(self, studio: StudioMod.ShapeStudio, spec: dict):
        self.s = studio
        self.spec = spec or {}
        self._old_phi_idx = self.s.which_arc_phi
        self._old_phi_pos = self.s.pos_phi
        self._old_amp_idx = self.s.which_arc_amp
        self._old_amp_pos = self.s.pos_amp
        self._old_k1 = self.s.bases[0].k
        self._old_k2 = self.s.bases[1].k
    def __enter__(self):
        if "phi" in self.spec and self.spec["phi"] is not None:
            idx, t = self.spec["phi"]; self.s.which_arc_phi, self.s.pos_phi = int(idx), float(t)
        if "amp" in self.spec and self.spec["amp"] is not None:
            idx, t = self.spec["amp"]; self.s.which_arc_amp, self.s.pos_amp = int(idx), float(t)
        if "freq" in self.spec and self.spec["freq"] is not None:
            k1, k2 = self.spec["freq"]; self.s.bases[0].k, self.s.bases[1].k = int(k1), int(k2)
        return self.s
    def __exit__(self, exc_type, exc, tb):
        self.s.which_arc_phi, self.s.pos_phi = self._old_phi_idx, self._old_phi_pos
        self.s.which_arc_amp, self.s.pos_amp = self._old_amp_idx, self._old_amp_pos
        self.s.bases[0].k, self.s.bases[1].k = self._old_k1, self._old_k2


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — Studio drives everything; we only collect
# ─────────────────────────────────────────────────────────────────────────────

class ShapeDeformDataset(Dataset):
    """
    Uses Studio's own resample + pairing path to produce any number of exemplars.
    - Class 0 specs from _exemplar_specs
    - Class 1 specs from _closest_specs  (mirrors Studio grid pairing)
    """

    @staticmethod
    def _theta_from_local_arc(cls: int, local_idx: int, t: float,
                              m_per_class: int, gap_frac: float) -> float:
        # class → global arc index: even for class 0, odd for class 1
        global_idx = 2 * int(local_idx) + (1 if cls == 1 else 0)
        total_arcs = 2 * int(m_per_class)
        base_w = 2.0 * math.pi / total_arcs
        gap = gap_frac * base_w
        usable = max(1e-9, base_w - gap)
        start = global_idx * base_w + gap / 2.0
        return (start + float(t) * usable) % (2.0 * math.pi)

    def __init__(self, nA, nB, image_size=128, intensity=1.0, bg=0.0, norm="max",
                 seed: Optional[int] = 42, batch: int = 8):
        self.H = self.W = int(image_size)
        self.intensity, self.bg, self.norm = float(intensity), float(bg), str(norm)
        self.batch = int(batch)

        N = int(nA) + int(nB)
        self.labels  = np.zeros(N, np.int64)
        self.images  = np.zeros((N, self.H, self.W), np.float32)

        # Diagnostics
        self.latents = np.zeros((N, 4), np.float32)  # (cos φ1, sin φ1, cos φ2, sin φ2)
        self.k1_used = np.zeros(N, np.int32)
        self.k2_used = np.zeros(N, np.int32)

        A_specs, B_specs = self._collect_specs_from_studio(nA, nB,independent_B=True)

        idx = 0
        for cls, specs in [(0, A_specs), (1, B_specs)]:
            for spec in specs:
                if not isinstance(spec, dict): spec = {}
                self.labels[idx] = cls

                # --- ring params from Studio (with safe fallbacks) ---
                m_phase = int(getattr(_STUDIO, "m_phase", 6))
                m_amp = int(getattr(_STUDIO, "m_amp", m_phase))
                gap_phase = float(getattr(_STUDIO, "gap_phase", getattr(_STUDIO, "gap_frac", 0.0)))
                gap_amp = float(getattr(_STUDIO, "gap_amp", getattr(_STUDIO, "gap_frac", 0.0)))

                # spec["phi"] and spec["amp"] are LOCAL arc indices (0..m-1) + position t∈[0,1]
                if isinstance(spec, dict) and (spec.get("phi") is not None):
                    i_phi, t_phi = int(spec["phi"][0]), float(spec["phi"][1])
                else:
                    i_phi, t_phi = 0, 0.5

                if isinstance(spec, dict) and (spec.get("amp") is not None):
                    i_amp, t_amp = int(spec["amp"][0]), float(spec["amp"][1])
                else:
                    i_amp, t_amp = 0, 0.5

                # class-aware mapping (even arcs for class 0, odd arcs for class 1)
                phi_theta = self._theta_from_local_arc(cls, i_phi, t_phi, m_phase, gap_phase)
                amp_theta = self._theta_from_local_arc(cls, i_amp, t_amp, m_amp, gap_amp)

                # Use BOTH phase and amplitude in the latent (so amp_theta is no longer “greyed out”)
                self.latents[idx] = [
                    math.cos(phi_theta), math.sin(phi_theta),
                    math.cos(amp_theta), math.sin(amp_theta),
                ]
                # Render via Studio then rasterize to fixed canvas (fit like Studio view)
                x, y, _, _ = _STUDIO._render_with_spec(cls, spec)

                # NEW: capture the ks that were actually used by Studio for this spec
                # BEFORE (wrong: reading after context restores 3/5)
                # x, y, _, _ = _STUDIO._render_with_spec(cls, spec)
                # self.k1_used[idx] = int(getattr(_STUDIO.bases[0], "k", 0))
                # self.k2_used[idx] = int(getattr(_STUDIO.bases[1], "k", 0))

                # AFTER (correct: read while the spec is applied and sampling is active)
                with _SpecApply(_STUDIO, spec):
                    x, y, _, _ = _STUDIO._render_with_spec(cls, spec)
                    k1_now = int(getattr(_STUDIO.bases[0], "k", 0))
                    k2_now = int(getattr(_STUDIO.bases[1], "k", 0))

                self.k1_used[idx] = k1_now
                self.k2_used[idx] = k2_now

                cx, cy = (self.W-1)/2.0, (self.H-1)/2.0
                rmax = float(np.max(np.hypot(x, y))); rmax = 1.0 if (not np.isfinite(rmax) or rmax<=1e-9) else rmax
                scale_px = 0.48 * min(self.H, self.W) / rmax
                img = _rasterize_filled_polygon(self.H, self.W, x, y, cx, cy, scale_px,
                                                fill_value=self.intensity, bg=self.bg)
                if self.norm == "max":
                    m = img.max(); 
                    if m > 0: img = img/m
                elif self.norm == "l2":
                    n = np.linalg.norm(img.reshape(-1)); 
                    if n > 0: img = img/n
                self.images[idx] = np.clip(img, 0.0, 1.0).astype(np.float32)
                idx += 1

        # Quick summary
        try:
            print(f"[v19] Unique k1: {np.unique(self.k1_used)}")
            print(f"[v19] Unique k2: {np.unique(self.k2_used)}")
        except Exception:
            pass






    def _collect_specs_from_studio(self, nA, nB, independent_B=True):
        """Ask Studio for batches until we have enough specs; robust to headless mode."""
        A: List[dict] = []; B: List[dict] = []
        a_needed, b_needed = int(nA), int(nB)

        def _resample_and_sync():
            if not hasattr(_STUDIO, "resample_exemplars"):
                raise RuntimeError("Studio lacks resample_exemplars(). Use the provided Studio file.")
            _STUDIO.resample_exemplars()
            try:
                if hasattr(_STUDIO, "_update_exemplar_views_impl"):
                    _STUDIO._update_exemplar_views_impl()
            except Exception: pass
            try:
                for _ in range(2): _QTAPP.processEvents()
            except Exception: pass

        guard = 0
        while len(A) < a_needed or len(B) < b_needed:
            _resample_and_sync()
            ex = getattr(_STUDIO, "_exemplar_specs", None) or []
            nb = getattr(_STUDIO, "_closest_specs", None) or []
            A.extend([s for s in ex if isinstance(s, dict)])
            if independent_B:
                B.extend([s for s in ex if isinstance(s, dict)])  # <— independent B
            else:
                B.extend([s for s in nb if isinstance(s, dict)])  # <— paired B
        return A[:a_needed], B[:b_needed]

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return torch.from_numpy(self.images[i])[None], torch.tensor(self.labels[i])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics & Plots
# ─────────────────────────────────────────────────────────────────────────────

def print_latent_distance_stats(dataset: "ShapeDeformDataset", normalize=True):
    Z = torch.from_numpy(dataset.latents.copy())
    labs = dataset.labels
    A = Z[labs==0]; B = Z[labs==1]
    iuA = torch.triu_indices(A.size(0), A.size(0), offset=1)
    iuB = torch.triu_indices(B.size(0), B.size(0), offset=1)
    DwA = euclidean_distance_matrix(A, normalize=normalize)[iuA[0], iuA[1]].mean().item() if iuA.numel() else 0.0
    DwB = euclidean_distance_matrix(B, normalize=normalize)[iuB[0], iuB[1]].mean().item() if iuB.numel() else 0.0
    Db  = euclidean_distance_matrix(A, B, normalize=normalize).mean().item()
    print(f"=== Latent-space *SQUARED* Euclidean distances ({'unit-sphere' if normalize else 'raw'}) ===")
    print(f"Within Class 0: {DwA:.4f}")
    print(f"Within Class 1: {DwB:.4f}")
    print(f"Overall within: {0.5*(DwA+DwB):.4f}")
    print(f"Between (0 vs 1): {Db:.4f}\n")


def print_image_distance_stats(dataset: "ShapeDeformDataset"):
    imgs = dataset.images.reshape(len(dataset), -1).astype(np.float32)
    A = torch.from_numpy(imgs[dataset.labels==0].copy())
    B = torch.from_numpy(imgs[dataset.labels==1].copy())
    iuA = torch.triu_indices(A.size(0), A.size(0), offset=1)
    iuB = torch.triu_indices(B.size(0), B.size(0), offset=1)
    DwA = euclidean_distance_matrix(A, normalize=True)[iuA[0], iuA[1]].mean().item() if iuA.numel() else 0.0
    DwB = euclidean_distance_matrix(B, normalize=True)[iuB[0], iuB[1]].mean().item() if iuB.numel() else 0.0
    Db  = euclidean_distance_matrix(A, B, normalize=True).mean().item()
    print("=== Pixel-space *SQUARED* Euclidean distances (L2-normalized) ===")
    print(f"Within Class 0: {DwA:.4f}")
    print(f"Within Class 1: {DwB:.4f}")
    print(f"Overall within: {0.5*(DwA+DwB):.4f}")
    print(f"Between (0 vs 1): {Db:.4f}\n")


def plot_phase_amp_torus_v13(dataset):
    """
    v13-style Phase–Amplitude Torus plot.
    Expects dataset.latents[:,4] = [cos φ1, sin φ1, cos χ, sin χ]
    and dataset.labels (0 or 1).
    """
    # --- Extract data
    Z: np.ndarray = np.asarray(dataset.latents, dtype=np.float32)
    labels: np.ndarray = np.asarray(dataset.labels, dtype=np.int64)

    # --- Recover angles from latent
    phi = (np.arctan2(Z[:, 1], Z[:, 0]) + 2 * np.pi) % (2 * np.pi)
    chi = (np.arctan2(Z[:, 3], Z[:, 2]) + 2 * np.pi) % (2 * np.pi)

    # --- Create plot
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

    # Fixed colors (v13 style)
    color_class0 = "green"   # dark green
    color_class1 = "lime"   # lime

    ax.scatter(phi[labels == 0], chi[labels == 0], s=10, c=color_class0, alpha=0.7, label="Class 0")
    ax.scatter(phi[labels == 1], chi[labels == 1], s=10, c=color_class1, alpha=0.7, label="Class 1")

    # --- Axes, labels, grid
    ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    labels_ticks = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels_ticks)
    ax.set_yticks(ticks); ax.set_yticklabels(labels_ticks)

    ax.set_xlabel("Phase angle φ₁", fontsize=11)
    ax.set_ylabel("Amplitude angle χ", fontsize=11)
    ax.set_title("Phase–Amplitude Torus (φ₁ vs χ)", fontsize=13)

    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="center")

    plt.tight_layout()
    plt.show()

# --- v13 helper (minimal re-introduction) ---
def _phase_angles_from_latents(dataset):
    Z = np.asarray(dataset.latents, dtype=np.float32)
    phi1 = np.arctan2(Z[:, 1], Z[:, 0])
    phi2 = np.arctan2(Z[:, 3], Z[:, 2])  # kept for API parity; not used here
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    phi2 = (phi2 + 2*np.pi) % (2*np.pi)
    return phi1, phi2

def plot_phase_ring_with_amp_bands(dataset: "ShapeDeformDataset", band_gap=0.15, alpha=0.8):
    labs = dataset.labels
    phi1, _ = _phase_angles_from_latents(dataset)
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    X = np.stack([np.cos(phi1), np.sin(phi1)], axis=1)

    # χ from dataset (prefer amp_angles/amp_latents; else use latents[:,2:4])
    if hasattr(dataset, "amp_angles") and np.any(getattr(dataset, "amp_angles", 0) != 0):
        chi = (dataset.amp_angles + 2*np.pi) % (2*np.pi)
    else:
        a = getattr(dataset, "amp_latents", None)
        if a is None:
            Z = np.asarray(dataset.latents, dtype=np.float32)
            a = Z[:, 2:4]                         # <— minimal adaptation
        chi = (np.arctan2(a[:,1], a[:,0]) + 2*np.pi) % (2*np.pi)

    # number of amp arcs per class — if not on dataset, fall back to Studio
    m_amp = getattr(dataset, "amp_m_arcs_per_class", None)
    if not m_amp:
        m_amp = int(getattr(_STUDIO, "m_amp", 2))   # <— minimal adaptation
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

    # v13 colors & markers (green/lime; dot for even, 'x' for odd)
    ax.scatter(XY[(labs==0) &  even,0], XY[(labs==0) &  even,1], s=12, alpha=alpha, color="green", label="C0 (even χ-arc)")
    ax.scatter(XY[(labs==0) & (~even),0], XY[(labs==0) & (~even),1], s=12, alpha=alpha, color="green", marker="x", label="C0 (odd χ-arc)")
    ax.scatter(XY[(labs==1) &  even,0], XY[(labs==1) &  even,1], s=12, alpha=alpha, color="lime",  label="C1 (even χ-arc)")
    ax.scatter(XY[(labs==1) & (~even),0], XY[(labs==1) & (~even),1], s=12, alpha=alpha, color="lime",  marker="x", label="C1 (odd χ-arc)")

    ax.set_aspect("equal", "box")
    ax.set_title("Phase Ring with Amplitude Parity Bands (via χ)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper right")
    plt.show()

def show_examples_with_nearest(dataset: "ShapeDeformDataset", k: int = 8):
    imgs = dataset.images; labs = dataset.labels
    A_idx = np.where(labs==0)[0]; B_idx = np.where(labs==1)[0]
    k = min(k, len(A_idx), len(B_idx)); A_idx = A_idx[:k]
    def l2n(x): n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8; return x/n
    A_flat = l2n(imgs[A_idx].reshape(len(A_idx), -1))
    B_flat = l2n(imgs[B_idx].reshape(len(B_idx), -1))
    D = 1.0 - (A_flat @ B_flat.T)
    nn = np.argmin(D, axis=1)
    fig, axes = plt.subplots(2, k, figsize=(1.6*k, 3.4))
    for j in range(k):
        axes[0,j].imshow(imgs[A_idx[j]], cmap="gray", vmin=0, vmax=1); axes[0,j].axis("off"); axes[0,j].set_title("C0")
        axes[1,j].imshow(imgs[B_idx[nn[j]]], cmap="gray", vmin=0, vmax=1); axes[1,j].axis("off"); axes[1,j].set_title("C1⋆")
    plt.suptitle("Exemplars (top) & Closest Matches (bottom)"); plt.tight_layout(); plt.show()


def _save_gray_png(path: str, img: np.ndarray):
    from PIL import Image
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def export_xab_and_categorisation(
    dataset,
    out_root: Path,
    excel_out: Path,
    n_within=100,
    n_between=100,
    seed=123,
    *,
    space: str = "pixel",          # "pixel" | "latent_phase" | "latent_all"
    studio=None                    # optional; not required
):
    """
    Create 200 pairs (100 within, 100 between), save images under ./experimentFiles/gabors/testing/pair{i}/,
    and write xab.xlsx and categorisation.xlsx in excel_out. Paths in Excel are relative to project root ("./experimentFiles/...").

    Minimal Studio-aware tweaks:
      - 'space' selects the vector space for nearest-neighbor search:
          "pixel"        -> L2-normalized pixel vectors (original behavior)
          "latent_phase" -> 2D (cosφ, sinφ)
          "latent_all"   -> 4D (cosφ, sinφ, cosχ, sinχ)
      - If 'studio' is provided, nothing else changes; we still read from dataset.*.
    """
    import numpy as _np
    from pathlib import Path as _Path
    from PIL import Image as _Image
    import pandas as _pd

    rng = _np.random.default_rng(seed)

    # --- Select vectors for NN search (minimal toggle; default = pixel as before)
    imgs = _np.asarray(dataset.images, dtype=_np.float32)  # HxW, 0..1
    labs = _np.asarray(dataset.labels, dtype=_np.int64)

    if space == "pixel":
        # original behavior: use pixel vectors (L2-normalized later)
        vecs = imgs.reshape(len(imgs), -1)
    elif space == "latent_phase":
        Z = _np.asarray(dataset.latents, dtype=_np.float32)
        vecs = Z[:, :2]  # (cosφ, sinφ)
    elif space == "latent_all":
        Z = _np.asarray(dataset.latents, dtype=_np.float32)
        vecs = Z[:, :4]  # (cosφ, sinφ, cosχ, sinχ)
    else:
        raise ValueError(f"Unsupported 'space': {space}")

    idx0 = _np.where(labs == 0)[0]
    idx1 = _np.where(labs == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise RuntimeError("Both classes must be present.")

    def _l2norm(X, eps=1e-12):
        nrm = _np.linalg.norm(X, axis=1, keepdims=True)
        nrm = _np.maximum(nrm, eps)
        return X / nrm

    def _nearest(in_rows, pool_rows):
        """
        Nearest neighbors using cosine distance converted to squared Euclidean on the unit sphere:
           d^2 = 2 - 2 * cos
        This matches your original logic and is stable for pixel and latent vectors.
        """
        A = _l2norm(in_rows.astype(_np.float32))
        B = _l2norm(pool_rows.astype(_np.float32))
        cos = A @ B.T
        D2  = 2.0 - 2.0 * _np.clip(cos, -1.0, 1.0)
        nn  = D2.argmin(axis=1)
        return nn, D2[_np.arange(in_rows.shape[0]), nn]

    # WITHIN pairs (unchanged logic)
    within = []
    choose = rng.integers(0, 2, size=n_within)
    for t in range(n_within):
        if choose[t] == 0:
            i = int(rng.integers(0, len(idx0)))
            a = idx0[i]
            nn, _d = _nearest(vecs[a:a+1], vecs[idx0])
            j = int(nn[0])
            if idx0[j] == a and len(idx0) > 1:
                # second-closest fallback (unchanged)
                A = _l2norm(vecs[a:a+1])
                B = _l2norm(vecs[idx0])
                cos = A @ B.T; D2 = 2.0 - 2.0 * _np.clip(cos, -1.0, 1.0)
                order = _np.argsort(D2[0]); j = int(order[1])
            b = idx0[j]
            within.append((a, b))
        else:
            i = int(rng.integers(0, len(idx1)))
            a = idx1[i]
            nn, _d = _nearest(vecs[a:a+1], vecs[idx1])
            j = int(nn[0])
            if idx1[j] == a and len(idx1) > 1:
                A = _l2norm(vecs[a:a+1])
                B = _l2norm(vecs[idx1])
                cos = A @ B.T; D2 = 2.0 - 2.0 * _np.clip(cos, -1.0, 1.0)
                order = _np.argsort(D2[0]); j = int(order[1])
            b = idx1[j]
            within.append((a, b))

    # BETWEEN pairs (unchanged logic)
    between = []
    choose = rng.integers(0, 2, size=n_between)
    for t in range(n_between):
        if choose[t] == 0:
            i = int(rng.integers(0, len(idx0)))
            a = idx0[i]
            nn, _d = _nearest(vecs[a:a+1], vecs[idx1])
            b = idx1[int(nn[0])]
            between.append((a, b))
        else:
            i = int(rng.integers(0, len(idx1)))
            a = idx1[i]
            nn, _d = _nearest(vecs[a:a+1], vecs[idx0])
            b = idx0[int(nn[0])]
            between.append((a, b))

    # --- saving images (unchanged)
    def _save_img(gray, path: _Path):
        arr = (_np.clip(gray, 0.0, 1.0) * 255).astype('uint8')
        im  = _Image.fromarray(arr, mode="L")
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, quality=95)

    out_root = _Path(out_root)
    excel_out = _Path(excel_out); excel_out.mkdir(parents=True, exist_ok=True)

    X, A_list, B_list, XPOSI = [], [], [], []
    def fname(global_idx: int, lab: int):
        return f"gabor_{global_idx:05d}_cat_{lab}.jpg"

    pair_id = 0
    for group in [within, between]:
        for i1, i2 in group:
            pair_dir = out_root / f"pair{pair_id}"
            pA = pair_dir / fname(i1, int(labs[i1]))
            pB = pair_dir / fname(i2, int(labs[i2]))
            _save_img(imgs[i1], pA)
            _save_img(imgs[i2], pB)
            if rng.random() < 0.5:
                X.append(str(pA)); XPOSI.append(0)
            else:
                X.append(str(pB)); XPOSI.append(0)
            A_list.append(str(pA)); B_list.append(str(pB))
            pair_id += 1

    # --- Excel outputs (unchanged, but robust)
    _proj_root = _Path(__file__).resolve().parent.parent

    def _to_rel_from_root(p):
        try:
            rel = _Path(p).resolve().relative_to(_proj_root)
        except Exception:
            rel = _Path(p)
        s = str(rel).replace('\\', '/')
        if not s.startswith("./"):
            s = "./" + s
        return s

    X = [_to_rel_from_root(p) for p in X]
    A_list = [_to_rel_from_root(p) for p in A_list]
    B_list = [_to_rel_from_root(p) for p in B_list]
    xab = _pd.DataFrame({"X": X, "A": A_list, "B": B_list, "XPOSI": XPOSI})

    all_imgs = A_list + B_list

    def lab_from_path(p: str) -> int:
        stem = _Path(p).stem  # gabor_00000_cat_0
        return int(stem.split("_")[-1])

    cats = ['l' if lab_from_path(p) == 1 else 'k' for p in all_imgs]
    ctrl = ['l' if rng.random() < 0.5 else 'k' for _ in all_imgs]
    cat = _pd.DataFrame({"Image_file": all_imgs, "category": cats, "control": ctrl})

    xab.to_excel(excel_out / "xab.xlsx", index=False)
    cat.to_excel(excel_out / "categorisation.xlsx", index=False)
    print(f"Wrote {(excel_out / 'xab.xlsx')} and {(excel_out / 'categorisation.xlsx')} (images under {out_root})")


# ─────────────────────────────────────────────────────────────────────────────
# Studio FULL configuration + hard-coded main
# ─────────────────────────────────────────────────────────────────────────────

def set_studio_from_main(
    # --- ring topology & global params ---
    m_phase=6, m_amp=6, gap=0.25, phase_deg=0.0,
    which_arc_phi=0, pos_phi=0.5, which_arc_amp=0, pos_amp=0.5,
    R=5.0, profile="absolute", sharp=0.3, amp_min=None, amp_max=None,
    k_max=8, m_freq=6, gap_freq=0.50,

    # --- sources (dropdowns) ---
    phase_src="None (ring)", amp_src="None (ring)", freq_src="None (ring)",

    # --- Base 1 ---
    k1=3, a1=0.9, phi1_deg=45.0,
    phase_mode1="signed_absolute", sphi1=0.25, Kphi1=10.0,
    amp_mode1="relative",      sA1=0.25,  KA1=10.0,

    # --- Base 2 ---
    k2=5, a2=0.6, phi2_deg=0.0,
    phase_mode2="signed_absolute", sphi2=0.25, Kphi2=10.0,
    amp_mode2="relative",      sA2=0.25,  KA2=10.0,

    # --- tiny harmonics list: (k, a, phi_deg, weight) ---
    tinys: List[Tuple[int,float,float,float]] = None
):
    # Global rings
    _STUDIO.m_phase=int(m_phase); _STUDIO.m_amp=int(m_amp)
    _STUDIO.gap=float(gap); _STUDIO.phase_deg=float(phase_deg)
    _STUDIO.which_arc_phi=int(which_arc_phi); _STUDIO.pos_phi=float(pos_phi)
    _STUDIO.which_arc_amp=int(which_arc_amp); _STUDIO.pos_amp=float(pos_amp)
    _STUDIO.R=float(R); _STUDIO.profile=str(profile); _STUDIO.sharp=float(sharp)
    if amp_min is not None: _STUDIO.amp_min=float(amp_min)
    if amp_max is not None: _STUDIO.amp_max=float(amp_max)

    # Frequency ring
    _STUDIO.k_max=int(k_max); _STUDIO.m_freq=int(m_freq); _STUDIO.gap_freq=float(gap_freq)

    # Sources (robust selection: text first, then index fallback if it contains "none")
    _STUDIO.cmb_phase_src.setCurrentText(str(phase_src))
    _STUDIO.cmb_amp_src.setCurrentText(str(amp_src))
    _STUDIO.cmb_freq_src.setCurrentText(str(freq_src))
    try:
        if "none" in str(phase_src).lower() and _STUDIO.cmb_phase_src.count()>1:
            _STUDIO.cmb_phase_src.setCurrentIndex(1)
    except Exception: pass
    try:
        if "none" in str(amp_src).lower() and _STUDIO.cmb_amp_src.count()>1:
            _STUDIO.cmb_amp_src.setCurrentIndex(1)
    except Exception: pass
    try:
        if "none" in str(freq_src).lower() and _STUDIO.cmb_freq_src.count()>1:
            _STUDIO.cmb_freq_src.setCurrentIndex(1)
    except Exception: pass

    # Base 1
    _STUDIO.bases[0].k=int(k1); _STUDIO.bases[0].a=float(a1); _STUDIO.bases[0].phi=math.radians(float(phi1_deg))
    _STUDIO.bases[0].pmode=str(phase_mode1); _STUDIO.bases[0].pstr=float(sphi1); _STUDIO.bases[0].pdiv=float(Kphi1)
    _STUDIO.bases[0].amode=str(amp_mode1);   _STUDIO.bases[0].astr=float(sA1);  _STUDIO.bases[0].adiv=float(KA1)

    # Base 2
    _STUDIO.bases[1].k=int(k2); _STUDIO.bases[1].a=float(a2); _STUDIO.bases[1].phi=math.radians(float(phi2_deg))
    _STUDIO.bases[1].pmode=str(phase_mode2); _STUDIO.bases[1].pstr=float(sphi2); _STUDIO.bases[1].pdiv=float(Kphi2)
    _STUDIO.bases[1].amode=str(amp_mode2);   _STUDIO.bases[1].astr=float(sA2);  _STUDIO.bases[1].adiv=float(KA2)

    # Tinys
    if tinys is not None:
        for i,(k,a,phi_deg,w) in enumerate(tinys):
            if i >= len(_STUDIO.tinys): break
            _STUDIO.tinys[i].k=int(k); _STUDIO.tinys[i].a=float(a)
            _STUDIO.tinys[i].phi=math.radians(float(phi_deg))
            _STUDIO.tinys[i].weight=float(w)

    _STUDIO.update_plots()


def main():
    # =====================
    # Hard-coded settings
    # =====================
    # Counts
    nA = 50   # class 0
    nB = nA   # class 1

    # Image/output settings
    image_size = 128
    intensity = 1.0
    bg = 0.0
    norm = "max"   # "max" | "l2" | "none"
    seed = 42
    batch_size = 64

    # Actions
    PRINT_STATS = False
    SHOW_EXAMPLES = True
    export_gabors = False


    # Studio preference (all sliders/knobs exposed exactly like older mains)
    set_studio_from_main(
        m_phase=6, m_amp=6, gap=0.1, phase_deg=0.0,
        which_arc_phi=0, pos_phi=0.5, which_arc_amp=0, pos_amp=0.5,
        R=5.0, profile="absolute", sharp=0.1, amp_min=None, amp_max=None,
        k_max=8, m_freq=6, gap_freq=0.1,
        phase_src="None (ring)", amp_src="None (ring)", freq_src="None (ring)",
        k1=3, a1=0.9, phi1_deg=45.0, phase_mode1="signed_absolute", sphi1=0.25, Kphi1=10.0,
        amp_mode1="relative", sA1=0.25, KA1=10.0,
        k2=5, a2=0.6, phi2_deg=0.0, phase_mode2="signed_absolute", sphi2=0.25, Kphi2=10.0,
        amp_mode2="relative", sA2=0.25, KA2=10.0,
        tinys=[(5, 1.0, 0.0, 1.0), (14, 0.0, 0.0, 0.8), (2, 0.0, 0.0, 0.3)]
    )
    # Ensure Studio has a valid, non-zero base state before specs are collected
    #_STUDIO.resample_exemplars()

    # Build dataset by asking Studio to generate batches until we have nA/nB
    ds = ShapeDeformDataset(nA=nA, nB=nB,
                            image_size=image_size,
                            intensity=intensity, bg=bg, norm=norm,
                            seed=seed, batch=500)

    # Split + loaders
    total = len(ds)
    val_sz = max(1, total // 5)   # 20%
    test_sz = max(1, total // 5)  # 20%
    train_sz = total - val_sz - test_sz
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [train_sz, val_sz, test_sz], generator=g)
    mk_loader = lambda d: DataLoader(d, batch_size=batch_size, shuffle=True, drop_last=False)
    trainloader, valloader, testloader = mk_loader(train_ds), mk_loader(val_ds), mk_loader(test_ds)

    # Stats/plots
    print_latent_distance_stats(ds)
    print_image_distance_stats(ds)
    show_examples_with_nearest(ds, k=min(8, nA, nB))

    # Export
    if export_gabors:
        print('Exporting XAB and categorisation files to ./experimentFiles/...')
        _script_dir = _Path(__file__).resolve().parent
        _project_root = _script_dir.parent  # sibling of src
        exp_root = _project_root / 'experimentFiles'
        # pixel-space nearest neighbors (original behavior)
        export_xab_and_categorisation(
            ds,
            exp_root / "gabors" / "testing",
            exp_root
        )
        # or, if you want latent-space nearest instead:
        #export_xab_and_categorisation(ds, exp_root / "gabors" / "testing", exp_root, space="latent_phase")  # φ only
        # export_xab_and_categorisation(ds, exp_root / "gabors" / "testing", exp_root, space="latent_all")  # φ & χ

        export_xab_and_categorisation(
            ds,
            exp_root / 'gabors' / 'testing',
            exp_root
        )
    plot_phase_ring_with_amp_bands(ds, band_gap=0.15)
    plot_phase_amp_torus_v13(ds)

if __name__ == "__main__":
    main()
