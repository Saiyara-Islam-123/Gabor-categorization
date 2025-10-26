
from __future__ import annotations

"""
shape_deform_dataset_v11_with_main.py
-------------------------------------

Version 11: fixes signature so that load_shape_deform_data(...) accepts
freq_* parameters (Py3.8-compatible typing), and forwards them into
ShapeDeformDataset. Otherwise identical layout/behavior to your v9 script,
plus the v10 frequency-ring feature.

- New knobs (already supported by the Dataset; now also by the loader & main):
  * freq_mode: "fixed" | "independent_uniform" | "ring_shared_arcs"
  * freq_m_arcs_per_class (int)
  * freq_gap_frac (float)
  * Other freq params (ring radius, scales, bounds, offset mode/strength/divisor)
"""

from typing import Optional, List, Tuple
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.path import Path
from pathlib import Path as _Path

# ─────────────────────────────────────────────────────────────────────────────
# Distance helpers
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

def _merge_alternating(arcs0: list, arcs1: list) -> list:
    arcs = []
    K0, K1 = len(arcs0), len(arcs1)
    for k in range(K0 + K1):
        if k % 2 == 0: arcs.append(arcs0[k // 2])
        else:          arcs.append(arcs1[k // 2])
    return arcs

def _sample_uniform_angle_on_arc(rng, start, end):
    if end < start:
        end += 2 * math.pi
    theta = rng.uniform(start, end)
    return theta % (2 * math.pi)

# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _shape_boundary(
    R: float,
    k1: int, k2: int,
    a1: float, a2: float,
    phi1: float, phi2: float,
    num_samples: int = 512,
    rot_rad: float = 0.0,
    radius_profile: str = "absolute",   # or "normalized"
    detail_terms: Optional[List[Tuple[int, float, float]]] = None  # [(k, a, phi)]
):
    theta = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    theta_rot = theta + rot_rad

    if radius_profile == "normalized":
        r = np.ones_like(theta_rot, dtype=np.float64) * R
        def add(term): r.__iadd__(R * term)
    elif radius_profile == "absolute":
        r = np.ones_like(theta_rot, dtype=np.float64) * R
        def add(term): r.__iadd__(term)
    else:
        raise ValueError("radius_profile must be 'normalized' or 'absolute'")

    if a1 != 0.0:
        add(a1 * np.cos(k1 * (theta_rot - phi1)))
    if a2 != 0.0:
        add(a2 * np.cos(k2 * (theta_rot - phi2)))

    if detail_terms:
        for k, a, phi in detail_terms:
            if a != 0.0:
                add(a * np.cos(k * (theta_rot - phi)))

    xs = r * np.cos(theta_rot)
    ys = r * np.sin(theta_rot)
    return xs, ys

def _rasterize_filled_polygon(H, W, xs, ys, cx, cy, scale_px,
                              fill_value=1.0, bg=0.0):
    from matplotlib.path import Path as _PathMP
    verts = np.stack([cx + scale_px*xs, cy + scale_px*ys], axis=1)
    poly = _PathMP(verts, closed=True)
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([xx + 0.5, yy + 0.5], axis=-1).reshape(-1, 2)
    mask = poly.contains_points(pts).reshape(H, W)
    img = np.full((H, W), bg, np.float32)
    img[mask] = fill_value
    return img

def _render_shape_image(
    H: int, W: int,
    ring_radius_norm: float,
    k1: int, k2: int,
    a1: float = 0.0, a2: float = 0.0,
    phi1: float = 0.0, phi2: float = 0.0,
    intensity: float = 1.0, bg: float = 0.0,
    norm: str = "max", global_rot_rad: float = 0.0,
    radius_profile: str = "absolute",
    amplitude_units: str = "pixels",
    detail_terms: Optional[List[Tuple[int, float, float]]] = None
):
    if radius_profile == "absolute" and amplitude_units == "pixels":
        R_px = float(ring_radius_norm)
        xs, ys = _shape_boundary(
            R=R_px, k1=k1, k2=k2,
            a1=float(a1), a2=float(a2),
            phi1=float(phi1), phi2=float(phi2),
            num_samples=512, rot_rad=global_rot_rad,
            radius_profile="absolute",
            detail_terms=detail_terms
        )
        scale_px = 1.0
    else:
        xs, ys = _shape_boundary(
            R=1.0, k1=k1, k2=k2,
            a1=float(a1), a2=float(a2),
            phi1=float(phi1), phi2=float(phi2),
            num_samples=512, rot_rad=global_rot_rad,
            radius_profile=radius_profile,
            detail_terms=detail_terms
        )
        scale_px = float(ring_radius_norm)

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    img = _rasterize_filled_polygon(H, W, xs, ys, cx, cy, scale_px,
                                    fill_value=float(intensity), bg=float(bg))
    if norm == "none":
        return img
    if norm == "max":
        m = img.max()
        if m > 0: img = img / m
        return np.clip(img, 0.0, 1.0)
    if norm == "l2":
        v = img.reshape(-1)
        n = np.linalg.norm(v)
        if n > 0: img = img / n
        return np.clip(img, 0.0, 1.0)
    return np.clip(img, 0.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _wrap_signed(x: float) -> float:
    return (x + math.pi) % (2*math.pi) - math.pi

class ShapeDeformDataset(Dataset):
    def __init__(self, nA, nB, m_arcs_per_class=2, gap_frac=0.2, phase_deg=0.0,
                 image_size=128, ring_radius_px=40,
                 k_lobes=3, k2_lobes=5,
                 a1=0.1, a2=0.1,
                 radius_profile="absolute", amplitude_units="pixels",
                 intensity=1.0, bg=0.0, seed=42,
                 norm="max", 
                 # amplitude modes / rings
                 amp_mode="ring_shared_arcs", amp_class_coupling="parity",
                 difficulty_sharp=0.0,
                 amp_ring_radius=0.15, amp_scale_a1=1.0, amp_scale_a2=1.0,
                 amp_m_arcs_per_class=None, amp_gap_frac=None,
                 # phase controls / ring
                 phase_mode="independent", phase_fixed_values=(0.0, 0.0),
                 # unified list with weights/roles
                 harmonics: Optional[List[Tuple[Optional[int], float, float, float, str]]] = None,
                 # tiny energy
                 tiny_energy_balance: bool = False,
                 tiny_energy_target: Optional[float] = None,
                 phase_offset_strength: float = 0.0,
                 amp_offset_strength: float = 0.0,
                 phase_offset_mode: str = "relative",
                 phase_offset_divisor: float = 10.0,
                 amp_offset_mode: str = "relative",
                 amp_offset_divisor: float = 10.0,
                 # ─── Frequency ring controls ───
                 freq_mode: str = "fixed",                      # "fixed" | "independent_uniform" | "ring_shared_arcs"
                 freq_m_arcs_per_class: Optional[int] = None,   # default = m_arcs_per_class
                 freq_gap_frac: Optional[float] = None,         # default = gap_frac
                 freq_ring_radius: float = 1.0,                 # radius in (k1,k2) space before quantization
                 freq_scale_k1: float = 1.0, freq_scale_k2: float = 1.0,  # anisotropic scaling
                 k1_min: int = 2, k1_max: int = 6,             # integer bounds for k1
                 k2_min: int = 3, k2_max: int = 7,             # integer bounds for k2
                 freq_offset_mode: str = "relative",            # like phase/amp offsets
                 freq_offset_strength: float = 1.0,
                 freq_offset_divisor: float = 10.0):

        rng = np.random.default_rng(seed)
        self.H = self.W = int(image_size)
        self.ring_radius_px = float(ring_radius_px)
        self.k1, self.k2 = int(k_lobes), int(k2_lobes)
        self.a1_fixed, self.a2_fixed = float(a1), float(a2)
        self.radius_profile = str(radius_profile)
        self.amplitude_units = str(amplitude_units)
        self.intensity, self.bg, self.norm = float(intensity), float(bg), str(norm)
        self.amp_mode = str(amp_mode)
        self.amp_class_coupling = str(amp_class_coupling)
        self.difficulty_sharp = float(difficulty_sharp)
        self.amp_ring_radius = float(amp_ring_radius)
        self.amp_scale_a1 = float(amp_scale_a1)
        self.amp_scale_a2 = float(amp_scale_a2)
        self.phase_mode = str(phase_mode)
        self.phase_fixed_values = (float(phase_fixed_values[0]), float(phase_fixed_values[1]))
        self.harmonics = harmonics or []
        self.tiny_energy_balance = bool(tiny_energy_balance)
        self.tiny_energy_target = tiny_energy_target
        self.phase_offset_strength = float(phase_offset_strength)
        self.amp_offset_strength = float(amp_offset_strength)
        self.phase_offset_mode = str(phase_offset_mode)
        self.phase_offset_divisor = float(phase_offset_divisor if phase_offset_divisor != 0 else 1.0)
        self.amp_offset_mode = str(amp_offset_mode)
        self.amp_offset_divisor = float(amp_offset_divisor if amp_offset_divisor != 0 else 1.0)

        # ─── Frequency ring params ───
        self.freq_mode = str(freq_mode)
        self.freq_m_arcs_per_class = int(freq_m_arcs_per_class) if freq_m_arcs_per_class is not None else int(m_arcs_per_class)
        self.freq_gap_frac = float(freq_gap_frac) if freq_gap_frac is not None else float(gap_frac)
        self.freq_ring_radius = float(freq_ring_radius)
        self.freq_scale_k1 = float(freq_scale_k1)
        self.freq_scale_k2 = float(freq_scale_k2)
        self.k1_min, self.k1_max = int(k1_min), int(k1_max)
        self.k2_min, self.k2_max = int(k2_min), int(k2_max)
        self.freq_offset_mode = str(freq_offset_mode)
        self.freq_offset_strength = float(freq_offset_strength)
        self.freq_offset_divisor = float(freq_offset_divisor if freq_offset_divisor != 0 else 1.0)

        # determine base rows from harmonics
        base = [h for h in self.harmonics if len(h) >= 5 and str(h[4]).lower() == "base"]
        if len(base) >= 2:
            b1, b2 = base[0], base[1]
            k1_base = None if (b1[0] is None) else int(b1[0])
            k2_base = None if (b2[0] is None) else int(b2[0])
        else:
            k1_base = int(self.k1)
            k2_base = int(self.k2)

        # amplitude arcs
        if amp_m_arcs_per_class is None:
            amp_m_arcs_per_class = m_arcs_per_class
        if amp_gap_frac is None:
            amp_gap_frac = gap_frac
        self.amp_m_arcs_per_class = int(amp_m_arcs_per_class)
        self.amp_gap_frac = float(amp_gap_frac)

        # Phase arcs
        phase_rad = math.radians(phase_deg)
        arcs_A, arcs_B, _ = _alternating_arcs_boundaries(m_arcs_per_class, phase_rad, gap_frac)

        # Amplitude arcs (for parity / labeling)
        amp_arcs_A, amp_arcs_B, _ = _alternating_arcs_boundaries(self.amp_m_arcs_per_class, 0.0, self.amp_gap_frac)
        amp_arcs_all = _merge_alternating(amp_arcs_A, amp_arcs_B)

        def _sample_amp_ring_shared(n, arcs_all, rng):
            j = rng.integers(0, len(arcs_all), size=n)
            chi = np.empty(n, dtype=np.float32)
            for i in range(n):
                s, e = arcs_all[j[i]]
                chi[i] = _sample_uniform_angle_on_arc(rng, s, e)
            amps_unit = np.stack([np.cos(chi), np.sin(chi)], axis=1).astype(np.float32)
            amps = np.stack([self.amp_ring_radius * self.amp_scale_a1 * amps_unit[:,0],
                             self.amp_ring_radius * self.amp_scale_a2 * amps_unit[:,1]], axis=1).astype(np.float32)
            return amps, j.astype(np.int32), chi

        # Amplitude selection
        if self.amp_mode == "fixed":
            A_amp = np.full((nA, 2), [self.a1_fixed, self.a2_fixed], np.float32)
            B_amp = np.full((nB, 2), [self.a1_fixed, self.a2_fixed], np.float32)
            A_amp_idx = np.zeros(nA, np.int32); B_amp_idx = np.zeros(nB, np.int32)
            A_amp_ang = np.zeros(nA, np.float32); B_amp_ang = np.zeros(nB, np.float32)
        elif self.amp_mode == "independent_uniform":
            A_amp = np.random.default_rng().uniform(-abs(self.a1_fixed), abs(self.a1_fixed), (nA, 2)).astype(np.float32)
            B_amp = np.random.default_rng().uniform(-abs(self.a1_fixed), abs(self.a1_fixed), (nB, 2)).astype(np.float32)
            A_amp_ang = np.arctan2(A_amp[:,1], A_amp[:,0]).astype(np.float32) % (2*np.pi)
            B_amp_ang = np.arctan2(B_amp[:,1], B_amp[:,0]).astype(np.float32) % (2*np.pi)
            total_amp_arcs = 2 * self.amp_m_arcs_per_class
            arc_w = 2*np.pi / total_amp_arcs
            A_amp_idx = np.floor(A_amp_ang / arc_w).astype(np.int32) % total_amp_arcs
            B_amp_idx = np.floor(B_amp_ang / arc_w).astype(np.int32) % total_amp_arcs
        elif self.amp_mode == "ring_shared_arcs":
            A_amp, A_amp_idx, A_amp_ang = _sample_amp_ring_shared(nA, amp_arcs_all, np.random.default_rng())
            B_amp, B_amp_idx, B_amp_ang = _sample_amp_ring_shared(nB, amp_arcs_all, np.random.default_rng())
        else:
            raise ValueError("invalid amp_mode")

        # Phase selection
        def _sample_phase_with_index(n, arcs_A, arcs_B, rng):
            thetas_A = np.empty(n, dtype=np.float32)
            idx_A = np.empty(n, dtype=np.int32)
            for t in range(n):
                i_local = rng.integers(len(arcs_A))
                s, e = arcs_A[i_local]
                th = _sample_uniform_angle_on_arc(rng, s, e)
                thetas_A[t] = th
                idx_A[t] = 2 * i_local  # even
            thetas_B = np.empty(n, dtype=np.float32)
            idx_B = np.empty(n, dtype=np.int32)
            for t in range(n):
                i_local = rng.integers(len(arcs_B))
                s, e = arcs_B[i_local]
                th = _sample_uniform_angle_on_arc(rng, s, e)
                thetas_B[t] = th
                idx_B[t] = 2 * i_local + 1  # odd
            return thetas_A, idx_A, thetas_B, idx_B

        if self.phase_mode == "fixed":
            phi1_A = np.full(nA, self.phase_fixed_values[0], np.float32)
            phi2_A = np.full(nA, self.phase_fixed_values[1], np.float32)
            phi1_B = np.full(nB, self.phase_fixed_values[0], np.float32)
            phi2_B = np.full(nB, self.phase_fixed_values[1], np.float32)
            i_phase_A = np.zeros(nA, np.int32); i_phase_B = np.zeros(nB, np.int32)
        else:
            phi1_A, i_phase_A, phi1_B, i_phase_B = _sample_phase_with_index(nA, arcs_A, arcs_B, np.random.default_rng())
            phi2_A, _,       phi2_B, _       = _sample_phase_with_index(nA, arcs_A, arcs_B, np.random.default_rng())

        # Labels
        if self.phase_mode == "fixed":
            A_lbl = (A_amp_idx % 2).astype(np.int32)
            B_lbl = (B_amp_idx % 2).astype(np.int32)
        else:
            if self.amp_class_coupling == "parity":
                A_lbl = ((i_phase_A + A_amp_idx) & 1).astype(np.int32)
                B_lbl = ((i_phase_B + B_amp_idx) & 1).astype(np.int32)
            else:
                A_lbl = (i_phase_A & 1).astype(np.int32)
                B_lbl = (i_phase_B & 1).astype(np.int32)

        labels = np.concatenate([A_lbl, B_lbl], 0)

        self.latents = np.stack([
            np.cos(np.concatenate([phi1_A, phi1_B])),
            np.sin(np.concatenate([phi1_A, phi1_B])),
            np.cos(np.concatenate([phi2_A, phi2_B])),
            np.sin(np.concatenate([phi2_A, phi2_B])),
        ], 1).astype(np.float32)
        self.latents_A = np.stack([np.cos(phi1_A), np.sin(phi1_A)], axis=1).astype(np.float32)
        self.latents_B = np.stack([np.cos(phi1_B), np.sin(phi1_B)], axis=1).astype(np.float32)

        if 'A_amp_ang' in locals() and 'B_amp_ang' in locals():
            self.amp_angles = np.concatenate([A_amp_ang, B_amp_ang], 0)
            self.amp_latents = np.concatenate([A_amp, B_amp], 0)
        else:
            self.amp_angles = np.zeros(nA+nB, np.float32)
            self.amp_latents = np.concatenate([A_amp, B_amp], 0)

        # ─────────────────────────────────────────────────────────────────────
        # Frequency ring sampling (k1,k2) per exemplar
        # ─────────────────────────────────────────────────────────────────────
        freq_arcs_A, freq_arcs_B, _ = _alternating_arcs_boundaries(self.freq_m_arcs_per_class, 0.0, self.freq_gap_frac)
        freq_arcs_all = _merge_alternating(freq_arcs_A, freq_arcs_B)

        def _sample_freq_ring_shared(n, arcs_all, rng):
            j = rng.integers(0, len(arcs_all), size=n)
            chi = np.empty(n, dtype=np.float32)
            for i in range(n):
                s, e = arcs_all[j[i]]
                chi[i] = _sample_uniform_angle_on_arc(rng, s, e)
            v = np.stack([np.cos(chi), np.sin(chi)], axis=1).astype(np.float32)
            K = np.stack([self.freq_ring_radius * self.freq_scale_k1 * v[:,0],
                          self.freq_ring_radius * self.freq_scale_k2 * v[:,1]], axis=1).astype(np.float32)
            return K, j.astype(np.int32), chi

        def _quantize_k_pair(K_cont):
            k1c = K_cont[:,0]; k2c = K_cont[:,1]
            k1_mid = 0.5 * (self.k1_min + self.k1_max)
            k2_mid = 0.5 * (self.k2_min + self.k2_max)
            k1_half = max(1.0, 0.5 * (self.k1_max - self.k1_min))
            k2_half = max(1.0, 0.5 * (self.k2_max - self.k2_min))
            k1 = np.rint(k1_mid + k1_half * k1c).astype(np.int32)
            k2 = np.rint(k2_mid + k2_half * k2c).astype(np.int32)
            k1 = np.clip(k1, self.k1_min, self.k1_max)
            k2 = np.clip(k2, self.k2_min, self.k2_max)
            return k1, k2

        treat_as_uniform = False
        if len(base) >= 2 and self.freq_mode == "fixed":
            if (base[0][0] is None) and (base[1][0] is None):
                treat_as_uniform = True

        if self.freq_mode == "fixed" and (k1_base is not None and k2_base is not None):
            K_A = np.full((nA, 2), [k1_base, k2_base], np.float32)
            K_B = np.full((nB, 2), [k1_base, k2_base], np.float32)
            (k1A_i,k2A_i) =(K_A[:,0] ,K_A[:,1])
            (k1B_i,k2B_i) =(K_B[:,0] ,K_B[:,1])

        elif self.freq_mode == "independent_uniform" or treat_as_uniform:
            rng2 = np.random.default_rng()
            K_A = rng2.uniform(-1.0, 1.0, (nA, 2)).astype(np.float32)
            K_B = rng2.uniform(-1.0, 1.0, (nB, 2)).astype(np.float32)
            k1A_i, k2A_i = _quantize_k_pair(K_A) if 'K_A' in locals() else (k1A_i, k2A_i)
            k1B_i, k2B_i = _quantize_k_pair(K_B) if 'K_B' in locals() else (k1B_i, k2B_i)
        elif self.freq_mode == "ring_shared_arcs":
            rng3 = np.random.default_rng()
            K_A, _, _ = _sample_freq_ring_shared(nA, freq_arcs_all, rng3)
            K_B, _, _ = _sample_freq_ring_shared(nB, freq_arcs_all, rng3)
            k1A_i, k2A_i = _quantize_k_pair(K_A) if 'K_A' in locals() else (k1A_i, k2A_i)
            k1B_i, k2B_i = _quantize_k_pair(K_B) if 'K_B' in locals() else (k1B_i, k2B_i)
        else:
            raise ValueError("invalid freq_mode")

        # Save per-exemplar ks for debugging/inspection
        self.k1_used = np.concatenate([k1A_i, k1B_i]).astype(np.int32)
        self.k2_used = np.concatenate([k2A_i, k2B_i]).astype(np.int32)

        def _apply_k_offset(k_ring: float, k_base: Optional[int]) -> int:
            if k_base is None:
                return int(round(k_ring))
            if self.freq_offset_mode == "signed_absolute":
                return int(round(float(k_base) + float(k_ring) / self.freq_offset_divisor))
            else:
                return int(round((1.0 - self.freq_offset_strength)*float(k_base) + self.freq_offset_strength*float(k_ring)))

        imgs = []
        for i in range(nA + nB):
            if i < nA:
                phi1 = float(phi1_A[i]); phi2 = float(phi2_A[i])
                a1, a2 = float(A_amp[i,0]), float(A_amp[i,1])
                k1_i, k2_i = int(k1A_i[i]), int(k2A_i[i])
            else:
                j = i - nA
                phi1 = float(phi1_B[j]); phi2 = float(phi2_B[j])
                a1, a2 = float(B_amp[j,0]), float(B_amp[j,1])
                k1_i, k2_i = int(k1B_i[j]), int(k2B_i[j])

            if len(base) >= 2 and self.freq_mode != "fixed":
                k1_i = _apply_k_offset(k1_i, k1_base)
                k2_i = _apply_k_offset(k2_i, k2_base)

                b1, b2 = base[0], base[1]
                if b1[2] is not None:
                    if self.phase_offset_mode == "signed_absolute":
                        phi1 = float(b1[2]) + _wrap_signed(phi1) / self.phase_offset_divisor
                    else:
                        d = _wrap_signed(phi1 - float(b1[2]))
                        phi1 = float(b1[2]) + self.phase_offset_strength * d
                if b2[2] is not None:
                    if self.phase_offset_mode == "signed_absolute":
                        phi2 = float(b2[2]) + _wrap_signed(phi2) / self.phase_offset_divisor
                    else:
                        d = _wrap_signed(phi2 - float(b2[2]))
                        phi2 = float(b2[2]) + self.phase_offset_strength * d

                if b1[1] is not None:
                    if self.amp_offset_mode == "signed_absolute":
                        a1 = float(b1[1]) + (a1 / self.amp_offset_divisor)
                    else:
                        a1 = (1.0 - self.amp_offset_strength) * float(b1[1]) + self.amp_offset_strength * a1
                if b2[1] is not None:
                    if self.amp_offset_mode == "signed_absolute":
                        a2 = float(b2[1]) + (a2 / self.amp_offset_divisor)
                    else:
                        a2 = (1.0 - self.amp_offset_strength) * float(b2[1]) + self.amp_offset_strength * a2

            lab = int(labels[i])
            amp_gain = self.difficulty_sharp if lab == 1 else 0.0
            a1_eff = a1 * (1.0 + amp_gain)
            a2_eff = a2 * (1.0 + amp_gain)

            detail: List[Tuple[int,float,float]] = []
            for row in self.harmonics:
                role = str(row[4]).lower() if len(row) >= 5 else "tiny"
                if role == "base":
                    continue
                k = int(row[0]); a = float(row[1]); phi = float(row[2])
                w = float(row[3]) if len(row) >= 4 else 0.0
                a_eff = a * (1.0 + amp_gain * w)
                if a_eff != 0.0:
                    detail.append((k, a_eff, phi))

            img = _render_shape_image(
                H=self.H, W=self.W,
                ring_radius_norm=self.ring_radius_px,
                k1=k1_i, k2=k2_i,
                a1=a1_eff, a2=a2_eff,
                phi1=phi1, phi2=phi2,
                intensity=self.intensity, bg=self.bg,
                norm=self.norm, global_rot_rad=0.0,
                radius_profile=self.radius_profile,
                amplitude_units=self.amplitude_units,
                detail_terms=detail
            )
            imgs.append(img)

        self.images = np.stack(imgs).astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return torch.from_numpy(self.images[i])[None], torch.tensor(self.labels[i])

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_latent_distance_stats(dataset: "ShapeDeformDataset", normalize=True):
    Z = dataset.latents.astype(np.float32)
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
    if hasattr(dataset, "amp_angles") and np.any(dataset.amp_angles != 0):
        chi = dataset.amp_angles.astype(np.float32)
        Z = np.stack([np.cos(chi), np.sin(chi)], axis=1).astype(np.float32)
    else:
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
    imgs = dataset.images.astype(np.float32)
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
# Visualization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _phase_angles_from_latents(dataset: "ShapeDeformDataset"):
    Z = dataset.latents
    phi1 = np.arctan2(Z[:,1], Z[:,0])
    phi2 = np.arctan2(Z[:,3], Z[:,2])
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    phi2 = (phi2 + 2*np.pi) % (2*np.pi)
    return phi1, phi2

def plot_latent_scatter(dataset: "ShapeDeformDataset", title="Latent ring by class"):
    labs = np.asarray(dataset.labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    t = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), lw=1.0, color="black", alpha=0.6)
    phi1, _ = _phase_angles_from_latents(dataset)
    X = np.stack([np.cos(phi1), np.sin(phi1)], axis=1)
    ax.scatter(X[labs==0,0], X[labs==0,1], s=10, color="green", alpha=0.7, label="Class 0")
    ax.scatter(X[labs==1,0], X[labs==1,1], s=10, color="lime",  alpha=0.7, label="Class 1")
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper right"); plt.show()

def plot_phase_amp_torus(dataset: "ShapeDeformDataset", alpha=0.6):
    labs = dataset.labels
    phi1, _ = _phase_angles_from_latents(dataset)
    if hasattr(dataset, "amp_angles") and np.any(dataset.amp_angles != 0):
        chi = dataset.amp_angles
    else:
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
    ax.grid(True, ls="--", alpha=0.3); ax.legend(); plt.show()

def plot_phase_ring_with_amp_bands(dataset: "ShapeDeformDataset", band_gap=0.15, alpha=0.8):
    labs = dataset.labels
    phi1, _ = _phase_angles_from_latents(dataset)
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    X = np.stack([np.cos(phi1), np.sin(phi1)], axis=1)

    if hasattr(dataset, "amp_angles") and np.any(dataset.amp_angles != 0):
        chi = (dataset.amp_angles + 2*np.pi) % (2*np.pi)
    else:
        a = dataset.amp_latents
        chi = (np.arctan2(a[:,1], a[:,0]) + 2*np.pi) % (2*np.pi)

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


# ─────────────────────────────────────────────────────────────────────────────
# Loader + export
# ─────────────────────────────────────────────────────────────────────────────

def load_shape_deform_data(
    nA=600, nB=600,
    image_size=128, ring_radius_px=40,
    radius_profile="absolute", amplitude_units="pixels",
    m_arcs_per_class=2, gap_frac=0.2, phase_deg=0.0,
    phase_mode="independent", phase_fixed_values=(0.0, 0.0),
    k1=3, k2=5, a1=8.0, a2=4.0,
    amp_mode="ring_shared_arcs", amp_class_coupling="parity",
    amp_ring_radius=0.15, amp_scale_a1=1.0, amp_scale_a2=1.0,
    amp_m_arcs_per_class=None, amp_gap_frac=None,
    difficulty_sharp=0.0,
    harmonics: Optional[List[Tuple[Optional[int], float, float, float, str]]] = None,
    tiny_energy_balance: bool = False,
    tiny_energy_target: Optional[float] = None,
    phase_offset_strength: float = 0.0,
    amp_offset_strength: float = 0.0,
    phase_offset_mode: str = "signed_absolute",
    phase_offset_divisor: float = 10.0,
    amp_offset_mode: str = "relative",
    amp_offset_divisor: float = 10.0,
    intensity=1.0, bg=0.0, norm="max",
    batch_size=256, seed=42,
    # --- Frequency (v11) ---
    freq_mode: str = "fixed",
    freq_m_arcs_per_class: Optional[int] = None,
    freq_gap_frac: Optional[float] = None,
    freq_ring_radius: float = 1.0,
    freq_scale_k1: float = 1.0,
    freq_scale_k2: float = 1.0,
    k1_min: int = 2, k1_max: int = 8,
    k2_min: int = 3, k2_max: int = 8,
    freq_offset_mode: str = "relative",
    freq_offset_strength: float = 1.0,
    freq_offset_divisor: float = 10.0,
):
    # derive k1,k2 from harmonics if base entries provide integers
    if harmonics:
        base = [h for h in harmonics if len(h) >= 5 and str(h[4]).lower() == "base"]
        if len(base) < 2:
            base = harmonics[:2]
        if len(base) >= 2 and base[0][0] is not None and base[1][0] is not None:
            k1, k2 = int(base[0][0]), int(base[1][0])

    ds = ShapeDeformDataset(
        nA=nA, nB=nB,
        m_arcs_per_class=m_arcs_per_class, gap_frac=gap_frac, phase_deg=phase_deg,
        image_size=image_size, ring_radius_px=ring_radius_px,
        k_lobes=k1, k2_lobes=k2,
        a1=a1, a2=a2,
        radius_profile=radius_profile, amplitude_units=amplitude_units,
        intensity=intensity, bg=bg, seed=seed, norm=norm,
        amp_mode=amp_mode, amp_class_coupling=amp_class_coupling,
        difficulty_sharp=difficulty_sharp,
        amp_ring_radius=amp_ring_radius, amp_scale_a1=amp_scale_a1, amp_scale_a2=amp_scale_a2,
        amp_m_arcs_per_class=amp_m_arcs_per_class, amp_gap_frac=amp_gap_frac,
        phase_mode=phase_mode, phase_fixed_values=phase_fixed_values,
        harmonics=harmonics,
        tiny_energy_balance=tiny_energy_balance,
        tiny_energy_target=tiny_energy_target,
        phase_offset_strength=phase_offset_strength,
        amp_offset_strength=amp_offset_strength,
        phase_offset_mode=phase_offset_mode,
        phase_offset_divisor=phase_offset_divisor,
        amp_offset_mode=amp_offset_mode,
        amp_offset_divisor=amp_offset_divisor,
        # frequency controls
        freq_mode=freq_mode,
        freq_m_arcs_per_class=freq_m_arcs_per_class,
        freq_gap_frac=freq_gap_frac,
        freq_ring_radius=freq_ring_radius,
        freq_scale_k1=freq_scale_k1,
        freq_scale_k2=freq_scale_k2,
        k1_min=k1_min, k1_max=k1_max,
        k2_min=k2_min, k2_max=k2_max,
        freq_offset_mode=freq_offset_mode,
        freq_offset_strength=freq_offset_strength,
        freq_offset_divisor=freq_offset_divisor,
    )
    # Basic split & loaders
    idx = np.arange(len(ds))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(0.7 * n); n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    class _Subset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices): self.base, self.idx = base_ds, indices
        def __len__(self): return len(self.idx)
        def __getitem__(self, i): return self.base[self.idx[i]]

    trainloader = DataLoader(_Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    valloader   = DataLoader(_Subset(ds, val_idx),   batch_size=batch_size, shuffle=False)
    testloader  = DataLoader(_Subset(ds, test_idx),  batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader, ds

def _save_gray_png(path: _Path, img: np.ndarray):
    from PIL import Image
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(str(path))

def export_xab_and_categorisation(ds: "ShapeDeformDataset", out_root: _Path, excel_out: _Path):
    import pandas as pd
    out_root = _Path(out_root)
    excel_out = _Path(excel_out)
    out_root.mkdir(parents=True, exist_ok=True)
    excel_out.mkdir(parents=True, exist_ok=True)

    A_idx = np.where(ds.labels == 0)[0]
    B_idx = np.where(ds.labels == 1)[0]
    A_list, B_list = [], []
    for i in A_idx:
        p = out_root / f"gabor_{i:05d}_cat_0.png"
        _save_gray_png(p, ds.images[i])
        A_list.append(str(p))
    for i in B_idx:
        p = out_root / f"gabor_{i:05d}_cat_1.png"
        _save_gray_png(p, ds.images[i])
        B_list.append(str(p))

    # Build XAB and categorisation sheets
    rng = np.random.default_rng(123)
    all_imgs = A_list + B_list

    def lab_from_path(p: str) -> int:
        stem = _Path(p).stem  # gabor_00000_cat_0
        return int(stem.split("_")[-1])

    cats = ['l' if lab_from_path(p) == 1 else 'k' for p in all_imgs]
    ctrl = ['l' if rng.random() < 0.5 else 'k' for _ in all_imgs]
    xab = pd.DataFrame({
        "A": A_list[:min(len(A_list), len(B_list))],
        "B": B_list[:min(len(A_list), len(B_list))],
    })
    cat = pd.DataFrame({"Image_file": all_imgs, "category": cats, "control": ctrl})
    xab.to_excel(excel_out / "xab.xlsx", index=False)
    cat.to_excel(excel_out / "categorisation.xlsx", index=False)
    print(f"Wrote {(excel_out / 'xab.xlsx')} and {(excel_out / 'categorisation.xlsx')} (images under {out_root})")

# ─────────────────────────────────────────────────────────────────────────────
# main() — v9 layout with added frequency block
# ─────────────────────────────────────────────────────────────────────────────

def main():
    export_gabors = False  # Set to False to skip file generation

    # Geometry & units
    image_size      = 128
    ring_radius_px  = 40
    radius_profile  = "absolute"    # "absolute" | "normalized"
    amplitude_units = "pixels"      # "pixels" | "relative"

    # Dataset sizes
    nA=50
    nB =nA

    # Ring topology
    m_arcs_per_class = 7
    gap_frac         = 0.30
    phase_deg        = 0.0
    phase_mode       = "independent"      # "independent" | "fixed"
    phase_fixed_values = (0.0, 0.0)       # used if phase_mode == "fixed"

    # Amplitude path for base pair
    amp_mode             = "fixed"  # "fixed" | "independent_uniform" | "ring_shared_arcs"
    amp_m_arcs_per_class = 7
    amp_gap_frac         = 0.10
    amp_ring_radius      = 2.0
    amp_scale_a1         = 0.5
    amp_scale_a2         = 0.5

    # Frequency path for base pair (v11; you can change these in place)
    freq_mode              = "ring_shared_arcs"  # "fixed" | "independent_uniform" | "ring_shared_arcs"
    freq_m_arcs_per_class  = 7
    freq_gap_frac          = 0.5

    # One unified list: (k, a, phi, weight, role)
    harmonics = [
        (None,   6, None, 0.0, "base"),  # base petals
        (None,  6, None, 0.0, "base"),  # secondary blobs
        (5,  1.4, 0.0, 0.8, "tiny"),  # micro ripples (70% of class gain)
        (12,  0, 0.0, 0.0, "tiny"),  # fine texture  (30% of class gain)
        (15,   0, 0.0, 0.1, "tiny"),  # elongation (class-agnostic)
    ]

    difficulty_sharp   = 0.0
    tiny_energy_balance = False
    tiny_energy_target  = sum(a*a for (k,a,phi,w,role) in harmonics if str(role).lower() != "base")

    # Ring-offset blending strength when base entries specify (a, phi)
    phase_offset_strength = 0.25
    amp_offset_strength   = 0.25

    # Signed–absolute phase offset control (used only if base φ provided)
    phase_offset_mode    = "signed_absolute"   # "relative" or "signed_absolute"
    phase_offset_divisor = 10.0                # K in φ_final = φ_base + signed(φ_sampled)/K

    # Signed–absolute amplitude offset control (used only if base amplitudes provided)
    amp_offset_mode    = "relative"           # "relative" or "signed_absolute"
    amp_offset_divisor = 10.0                  # K in a_final = a_base + a_ring/K

    trainloader, valloader, testloader, ds = load_shape_deform_data(
        nA=nA, nB=nB,
        image_size=image_size, ring_radius_px=ring_radius_px,
        radius_profile=radius_profile, amplitude_units=amplitude_units,
        m_arcs_per_class=m_arcs_per_class, gap_frac=gap_frac, phase_deg=phase_deg,
        phase_mode=phase_mode, phase_fixed_values=phase_fixed_values,
        amp_mode=amp_mode, amp_m_arcs_per_class=amp_m_arcs_per_class, amp_gap_frac=amp_gap_frac,
        amp_ring_radius=amp_ring_radius, amp_scale_a1=amp_scale_a1, amp_scale_a2=amp_scale_a2,
        difficulty_sharp=difficulty_sharp,
        harmonics=harmonics,
        tiny_energy_balance=tiny_energy_balance,
        tiny_energy_target=tiny_energy_target,
        phase_offset_strength=phase_offset_strength,
        amp_offset_strength=amp_offset_strength,
        phase_offset_mode=phase_offset_mode,
        phase_offset_divisor=phase_offset_divisor,
        amp_offset_mode=amp_offset_mode,
        amp_offset_divisor=amp_offset_divisor,
        batch_size=256, seed=42,
        # frequency params
        freq_mode=freq_mode,
        freq_m_arcs_per_class=freq_m_arcs_per_class,
        freq_gap_frac=freq_gap_frac,
    )

    print(f"Dataset size: {len(ds)} images  |  Image shape: {ds.images.shape[1:]}")
    print_latent_distance_stats(ds, normalize=True)
    print_amplitude_distance_stats(ds, normalize=False)
    print_image_distance_stats(ds)

    # Sanity: show first 10 frequencies actually used
    try:
        ks_preview = list(zip(ds.k1_used[:10], ds.k2_used[:10]))
        print("First 10 (k1,k2):", ks_preview)
    except Exception as e:
        print("(k1,k2) preview unavailable:", e)

    if export_gabors:
        print('Exporting XAB and categorisation files to ./experimentFiles/...')
        _script_dir = _Path(__file__).resolve().parent
        _project_root = _script_dir.parent  # sibling of src
        exp_root = _project_root / 'experimentFiles'
        export_xab_and_categorisation(
            ds,
            exp_root / 'gabors' / 'testing',
            exp_root
        )

    plot_latent_scatter(ds, title="Latent projection by class")
    plot_phase_amp_torus(ds)
    plot_phase_ring_with_amp_bands(ds, band_gap=0.15)
    show_examples(ds, n=12)


if __name__ == "__main__":
    main()
