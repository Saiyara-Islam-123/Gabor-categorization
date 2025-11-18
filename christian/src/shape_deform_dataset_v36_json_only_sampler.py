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

import json as _json

import io, contextlib

# JSON config captured from set_studio_from_main
_CFG = None
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

# --- Metrics diagnostics helpers (add near imports) ---

def _metrics_rng(seed: int):
    """Single RNG used for BOTH classes during metrics sampling."""
    return np.random.default_rng(int(seed))

from collections import Counter
import numpy as np
import math

# ─────────────────────────────────────────────────────────────────────────────
# Global-band helpers (v13-style)
# ─────────────────────────────────────────────────────────────────────────────
def global_band_index(theta, m, gap_frac):
    """Map angle theta ∈ [0,2π) to global band index 0..(2*m-1)."""
    theta = float(theta) % (2.0 * math.pi)
    base_w = 2.0 * math.pi / (2*int(m))
    gap = float(gap_frac) * base_w
    # band k spans: start = k*base_w + gap/2  ; usable width = base_w - gap
    # invert approximately by subtracting gap/2 then dividing by base_w
    k = int(math.floor((theta - gap/2.0) / base_w)) % (2*int(m))
    return k

def compute_global_bands_from_latents(dataset, gap):
    Z = np.asarray(dataset.latents, dtype=np.float32)
    phi = (np.arctan2(Z[:,1], Z[:,0]) + 2*np.pi) % (2*np.pi)
    chi = (np.arctan2(Z[:,3], Z[:,2]) + 2*np.pi) % (2*np.pi)
    # Pull m_phase/m_amp from config if available; else infer from coverage
    try:
        m_phase = int(_CFG.get('m_phase', 6))
        m_amp   = int(_CFG.get('m_amp', 6))
    except Exception:
        m_phase, m_amp = 6, 6
    pg = np.array([global_band_index(t, m_phase, gap) for t in phi], dtype=np.int32)
    ag = np.array([global_band_index(t, m_amp,   gap) for t in chi], dtype=np.int32)
    return phi, chi, pg, ag

def _log_sampling_stats(label: str, specs: list, m_phase: int = None, m_amp: int = None):
    """
    Logs per-class coverage over PHASE (φ) arcs and AMP (χ) bands.
    Supports either:
      - s["phi"] == (arc_idx, t) and s["amp"] == (band_idx, t), or
      - s["phi_rad"], s["chi_rad"] continuous angles in [0, 2π).
    """
    phase_idx, amp_idx, t_phi, t_amp = [], [], [], []

    for s in specs:
        # Try tuple form first
        if isinstance(s.get("phi"), (tuple, list)) and len(s["phi"]) == 2:
            phase_idx.append(int(s["phi"][0]))
            t_phi.append(float(s["phi"][1]))
        elif "phi_rad" in s:
            # Bin continuous angle into 2*m_phase global sectors if m_phase provided
            if m_phase is not None:
                total = 2 * int(m_phase)
                k = int(math.floor((s["phi_rad"] % (2*np.pi)) / (2*np.pi/total)))
                phase_idx.append(k)

        if isinstance(s.get("amp"), (tuple, list)) and len(s["amp"]) == 2:
            amp_idx.append(int(s["amp"][0]))
            t_amp.append(float(s["amp"][1]))
        elif "chi_rad" in s:
            if m_amp is not None:
                total = 2 * int(m_amp)
                k = int(math.floor((s["chi_rad"] % (2*np.pi)) / (2*np.pi/total)))
                amp_idx.append(k)

    print(f"[metrics] {label} — phase arcs:", dict(Counter(phase_idx)))
    print(f"[metrics] {label} — amp bands :", dict(Counter(amp_idx)))
    if t_phi:
        print(f"[metrics] {label} — t_phi mean/std:", float(np.mean(t_phi)), float(np.std(t_phi)))
    if t_amp:
        print(f"[metrics] {label} — t_amp mean/std:", float(np.mean(t_amp)), float(np.std(t_amp)))



def _sample_specs_for_class_with_studio(studio, cls: int, n: int, seed: int = 12345):
    import numpy as _np
    assert isinstance(_CFG, dict), "_CFG (JSON) not loaded"

    # Modes (JSON-only)
    phase_none = str(_CFG.get("phase_src", "")).startswith("None")
    amp_none   = str(_CFG.get("amp_src",   "")).startswith("None")
    freq_none  = str(_CFG.get("freq_src",  "")).startswith("None")

    # Ring sizes (JSON-only)
    m_phase = int(_CFG["m_phase"])
    m_amp   = int(_CFG["m_amp"])
    assert m_phase >= 1 and m_amp >= 1, "m_phase/m_amp must be >=1"

    # k-bounds (JSON-only)
    kmin1 = int(_CFG.get("k_min1", _CFG.get("kmin1", 2)))
    kmin2 = int(_CFG.get("k_min2", _CFG.get("kmin2", 3)))
    kmax  = int(_CFG.get("k_max",  _CFG.get("kmax",  8)))
    assert kmax >= kmin1 and kmax >= kmin2, "k_max must be >= k_min*"

    rng = _np.random.default_rng(seed + 7919 * int(cls))
    specs = []

    for _ in range(int(n)):
        spec = {"class_id": int(cls)}

        # φ: None(ring) => independent (i,t) draw, else fixed from JSON
        if phase_none:
            spec["phi"] = (int(rng.integers(0, m_phase)), float(rng.random()))
        else:
            spec["phi"] = (int(_CFG.get("which_arc_phi", 0)),
                           float(_CFG.get("pos_phi", 0.5)))

        # χ (amp): None(ring) => independent draw, else fixed from JSON
        if amp_none:
            spec["amp"] = (int(rng.integers(0, m_amp)), float(rng.random()))
        else:
            spec["amp"] = (int(_CFG.get("which_arc_amp", 0)),
                           float(_CFG.get("pos_amp", 0.5)))

        # freq: None(ring) => inclusive kmax, else fixed from JSON
        if freq_none:
            spec["freq"] = (
                int(rng.integers(min(kmin1, kmax), max(kmin1, kmax) + 1)),
                int(rng.integers(min(kmin2, kmax), max(kmin2, kmax) + 1)),
            )
        else:
            spec["freq"] = (int(_CFG.get("k1", kmin1)),
                            int(_CFG.get("k2", kmin2)))

        specs.append(spec)

    if len(specs) != int(n):
        raise RuntimeError(f"[sampler] produced {len(specs)} < n={n}")
    return specs


def _build_ds_for_stats(studio, n_per_class: int, seed: int = 12345):
    """
    Build the minimal dataset object that your existing print_* functions expect.
    This should mirror how the dataset is constructed for stats (no exports).
    """
    # If you already have a constructor/factory used by stats, call that here instead.
    # Otherwise, create your DS the same way your stats routines do:
    class _DS:
        pass
    ds = _DS()
    ds.studio = studio
    ds.n_per_class = n_per_class
    ds.specs0 = _sample_specs_for_class_with_studio(studio, 0, n_per_class, seed)
    ds.specs1 = _sample_specs_for_class_with_studio(studio, 1, n_per_class, seed) #seed +777
    # If your print_* need more fields, add them here (paths, config, etc.)
    return ds

def get_latent_and_image_stats_text_for_studio(studio, n_per_class: int = 24, seed: int = 12345):
    """
    Runs your existing print_latent_distance_stats(ds) and print_image_distance_stats(ds),
    captures their stdout, and returns a single formatted string.
    """
    ds = _build_ds_for_stats(studio, n_per_class, seed)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        # Your existing functions:
        print_latent_distance_stats(ds)
        print_image_distance_stats(ds)
    return buf.getvalue()


def load_studio_settings_json(p: str) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        cfg = _json.load(f)
        global _CFG
        _CFG = cfg
    return cfg

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

    def _collect_specs_from_json(self, nA, nB, independent_B=True):
        """JSON-only sampler: read _CFG and generate specs for A and B; no Studio dependency."""
        import numpy as np
        assert isinstance(_CFG, dict), "JSON config missing; pass cfg_path=... or place studio_settings.json next to this file."

        m_phase = int(_CFG.get('m_phase', 6)); m_amp = int(_CFG.get('m_amp', 6))
        kmin1 = int(_CFG.get('k_min1', _CFG.get('kmin1', 2)))
        kmin2 = int(_CFG.get('k_min2', _CFG.get('kmin2', 2)))
        kmax  = int(_CFG.get('k_max',  _CFG.get('kmax',  8)))

        phase_none = str(_CFG.get('phase_src', 'None')).lower().startswith('none')
        amp_none   = str(_CFG.get('amp_src',   'None')).lower().startswith('none')
        freq_none  = str(_CFG.get('freq_src',  'None')).lower().startswith('none')

        which_arc_phi = int(_CFG.get('which_arc_phi', 0)); pos_phi = float(_CFG.get('pos_phi', 0.5))
        which_arc_amp = int(_CFG.get('which_arc_amp', 0)); pos_amp = float(_CFG.get('pos_amp', 0.5))
        k1_def = int(_CFG.get('k1', kmin1)); k2_def = int(_CFG.get('k2', kmin2))

        def draw(n, seed_offset):
            r = np.random.default_rng(int(getattr(self, 'metrics_seed', 12345)) + seed_offset)
            out = []
            for _ in range(int(n)):
                if phase_none:
                    i_phi = int(r.integers(0, m_phase)); t_phi = float(r.random())
                else:
                    i_phi, t_phi = which_arc_phi, pos_phi
                if amp_none:
                    i_amp = int(r.integers(0, m_amp)); t_amp = float(r.random())
                else:
                    i_amp, t_amp = which_arc_amp, pos_amp
                if freq_none:
                    k1 = int(r.integers(min(kmin1, kmax), max(kmin1, kmax) + 1))
                    k2 = int(r.integers(min(kmin2, kmax), max(kmin2, kmax) + 1))
                else:
                    k1, k2 = k1_def, k2_def
                out.append({'phi': (i_phi, t_phi), 'amp': (i_amp, t_amp), 'freq': (k1, k2)})
            return out

        A_specs = draw(nA, seed_offset=17)
        B_specs = draw(nB, seed_offset=19) if independent_B else draw(nB, seed_offset=17)
        return A_specs, B_specs
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


    @staticmethod
    def _theta_from_local_arc_with_parity(parity: int, local_idx: int, t: float,
                                          m_per_class: int, gap_frac: float) -> float:
        """Angle for a given ring using an explicit global-band parity (0 even, 1 odd).
        This does NOT depend on class; parity is deterministic from local indices."""
        parity = int(parity) & 1
        total_arcs = 2 * int(m_per_class)
        base_w = 2.0 * math.pi / total_arcs
        gap = gap_frac * base_w
        usable = max(1e-9, base_w - gap)
        global_idx = 2 * int(local_idx) + parity
        start = global_idx * base_w + gap / 2.0
        return (start + float(t) * usable) % (2.0 * math.pi)

    def __init__(self, nA, nB, image_size=128, intensity=1.0, bg=0.0, norm="max", seed: Optional[int] = 42, batch: int = 8,independent_B: bool = True, metrics_seed: int = 12345, metrics_debug: bool = False, **kwargs):

        # v36: simplest JSON-only load — expect ./studio_settings.json relative to current working dir
        global _CFG
        try:
            load_studio_settings_json("./studio_settings.json")
        except Exception as e:
            raise AssertionError("studio_settings.json not found in current working directory; "
                                 "run from that folder or pass an absolute path.") from e


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

        # at dataset init, after you allocate arrays:
        self.specs = [None] * N  # one dict per sample


        self.independent_B = bool(independent_B)
        self.metrics_seed = int(metrics_seed)
        self.metrics_debug = bool(metrics_debug)

        A_specs, B_specs = self._collect_specs_from_json(nA, nB, independent_B=self.independent_B)

        assert len(A_specs) == int(nA) and len(B_specs) == int(nB), \
            f"[collect] got A={len(A_specs)}/{nA}, B={len(B_specs)}/{nB}"

        # NEW: sanity

        assert len(A_specs) == nA and len(B_specs) == nB, f"[collect] got A={len(A_specs)}/{nA}, B={len(B_specs)}/{nB}"
        assert all(isinstance(s, dict) for s in A_specs), "[collect] Class 0 has a None spec"
        assert all(isinstance(s, dict) for s in B_specs), "[collect] Class 1 has a None spec"

        idx = 0
        for cls, specs in [(0, A_specs), (1, B_specs)]:
            for spec in specs:
                # store raw spec and label
                self.specs[idx] = spec
                self.labels[idx] = cls

                # --- ring params from JSON (Studio is the source of truth) ---
                m_phase = int(_CFG['m_phase'])
                m_amp = int(_CFG['m_amp'])
                gap = float(_CFG.get('gap', 0.0))
                phase_deg = float(_CFG.get('phase_deg', 0.0))
                amp_min = float(_CFG.get('amp_min', 0.5))
                amp_max = float(_CFG.get('amp_max', 3.5))
                parity = bool(_CFG.get('parity_flip', False))

                # spec["phi"] and spec["amp"] are LOCAL arc indices (0..m-1) + position t∈[0,1]
                if isinstance(spec, dict) and (spec.get("phi") is not None):
                    i_phi, t_phi = int(spec["phi"][0]), float(spec["phi"][1])
                else:
                    i_phi, t_phi = 0, 0.5

                if isinstance(spec, dict) and (spec.get("amp") is not None):
                    i_amp, t_amp = int(spec["amp"][0]), float(spec["amp"][1])
                else:
                    i_amp, t_amp = 0, 0.5

                
                

                # Angles from JSON-only params using local (i,t) → θ mapping
                # v13-style: independent GLOBAL parities; label = XOR of GLOBAL indices
                _rngp = np.random.default_rng(self.metrics_seed + 100003 + idx)
                p_phi = int(_rngp.integers(0, 2))      # GLOBAL φ parity
                p_amp = int(_rngp.integers(0, 2))      # GLOBAL χ parity

                # φ ring with explicit global parity, then apply global phase_deg
                phi_theta = ShapeDeformDataset._theta_from_local_arc_with_parity(p_phi, i_phi, t_phi, m_phase, gap)
                phi_theta = (float(phi_theta) + math.radians(float(phase_deg))) % (2.0 * math.pi)

                # χ ring with explicit global parity
                amp_theta = ShapeDeformDataset._theta_from_local_arc_with_parity(p_amp, i_amp, t_amp, m_amp, gap)

                # Compute GLOBAL band indices and XOR label (checkerboard cell)
                phase_global = 2 * int(i_phi) + int(p_phi)
                amp_global   = 2 * int(i_amp) + int(p_amp)
                self.labels[idx] = int((phase_global + amp_global) & 1)
                phi_theta = float(phi_theta); amp_theta = float(amp_theta)

                # Latents: use phase AND amplitude (you had phi duplicated twice before)
                self.latents[idx] = [
                    math.cos(phi_theta), math.sin(phi_theta),
                    math.cos(amp_theta), math.sin(amp_theta),
                ]

                # Render via Studio, but spec is fully JSON-driven
                with _SpecApply(_STUDIO, spec):
                    x, y, _, _ = _STUDIO._render_with_spec(cls, spec)
                    k1_now = int(getattr(_STUDIO.bases[0], "k", 0))
                    k2_now = int(getattr(_STUDIO.bases[1], "k", 0))

                self.k1_used[idx] = k1_now
                self.k2_used[idx] = k2_now

                # Rasterize (unchanged)
                cx, cy = (self.W - 1) / 2.0, (self.H - 1) / 2.0
                rmax = float(np.max(np.hypot(x, y)))
                rmax = 1.0 if (not np.isfinite(rmax) or rmax <= 1e-9) else rmax
                scale_px = 0.48 * min(self.H, self.W) / rmax
                img = _rasterize_filled_polygon(
                    self.H, self.W, x, y, cx, cy, scale_px,
                    fill_value=self.intensity, bg=self.bg
                )
                if self.norm == "max":
                    m = img.max()
                    if m > 0: img = img / m
                elif self.norm == "l2":
                    n = np.linalg.norm(img.reshape(-1))
                    if n > 0: img = img / n

                self.images[idx] = np.clip(img, 0.0, 1.0).astype(np.float32)
                idx += 1

        # <-- the filled-count check MUST be here, AFTER the loops
        if idx != (nA + nB):
            raise RuntimeError(
                f"[dataset] Filled {idx} specs but expected {nA + nB}. "
                f"A_specs={len(A_specs)}, B_specs={len(B_specs)}"
            )

        # Quick summary (unchanged)
        try:
            print(f"[v19] Unique k1: {np.unique(self.k1_used)}")
            print(f"[v19] Unique k2: {np.unique(self.k2_used)}")
        except Exception:
            pass

    def _label_v13(spec, m_phase, m_amp, rng):
        # local indices from Studio spec (0..m-1)
        i_phi = int(spec.get("phi", (0, 0.5))[0])
        i_amp = int(spec.get("amp", (0, 0.5))[0])

        # draw *independent* parity bits the way v13 effectively does via alternating arcs
        p_phi = int(rng.integers(0, 2))  # even/odd phase band (global)
        p_amp = int(rng.integers(0, 2))  # even/odd amp band (global)

        phase_global = 2 * i_phi + p_phi  # in 0..(2*m_phase-1)
        amp_global = 2 * i_amp + p_amp  # in 0..(2*m_amp-1)
        return (phase_global + amp_global) & 1, phase_global, amp_global


    def _collect_specs_from_studio(self, nA, nB, independent_B=True):
        """
        JSON-only sampler: build specs directly from _CFG without consulting Studio.
        - Class 0 and Class 1 are sampled independently when independent_B=True.
        - Each spec is a dict: {'phi': (i_phi, t_phi), 'amp': (i_amp, t_amp), 'freq': (k1, k2)}
          where local indices i_* are in [0 .. m_* - 1] and t_* in [0,1].
        """
        import numpy as _np
        assert isinstance(_CFG, dict), "JSON config missing; pass cfg_path=... or place studio_settings.json next to this file."

        # JSON ring sizes + freq bounds
        m_phase = int(_CFG["m_phase"])
        m_amp = int(_CFG["m_amp"])
        kmin1 = int(_CFG.get("k_min1", _CFG.get("kmin1", 2)))
        kmin2 = int(_CFG.get("k_min2", _CFG.get("kmin2", 3)))
        kmax = int(_CFG.get("k_max", _CFG.get("kmax", 8)))
        if kmax < max(kmin1, kmin2):
            kmax = max(kmin1, kmin2)

        # Modes from JSON
        phase_none = str(_CFG.get("phase_src", "")).lower().startswith("none")
        amp_none = str(_CFG.get("amp_src", "")).lower().startswith("none")
        freq_none = str(_CFG.get("freq_src", "")).lower().startswith("none")

        rng0 = _np.random.default_rng(self.metrics_seed + 7919*0)
        rng1 = _np.random.default_rng(self.metrics_seed + 7919*1)

        def _draw_specs(rng, count, cls):
            out = []
            for _ in range(int(count)):
                if phase_none:
                    i_phi = int(rng.integers(0, m_phase))
                    t_phi = float(rng.random())
                else:
                    i_phi = int(_CFG.get("which_arc_phi", 0))
                    t_phi = float(_CFG.get("pos_phi", 0.5))

                if amp_none:
                    i_amp = int(rng.integers(0, m_amp))
                    t_amp = float(rng.random())
                else:
                    i_amp = int(_CFG.get("which_arc_amp", 0))
                    t_amp = float(_CFG.get("pos_amp", 0.5))

                if freq_none:
                    k1 = int(rng.integers(min(kmin1, kmax), max(kmin1, kmax) + 1))
                    k2 = int(rng.integers(min(kmin2, kmax), max(kmin2, kmax) + 1))
                else:
                    k1 = int(_CFG.get("k1", kmin1))
                    k2 = int(_CFG.get("k2", kmin2))

                out.append({"phi": (i_phi, t_phi), "amp": (i_amp, t_amp), "freq": (k1, k2)})
            return out

        A_specs = _draw_specs(rng0, nA, cls=0)

        if independent_B:
            B_specs = _draw_specs(rng1, nB, cls=1)
        else:
            # Simple mirrored pairing across the ring as a proxy for “closest”
            B_specs = []
            for s in A_specs[:nB]:
                i_phi, t_phi = s["phi"]
                i_amp, t_amp = s["amp"]
                k1, k2 = s["freq"]
                j_phi = (i_phi + m_phase // 2) % m_phase if m_phase > 1 else 0
                j_amp = (i_amp + m_amp // 2) % m_amp if m_amp > 1 else 0
                B_specs.append({"phi": (j_phi, t_phi), "amp": (j_amp, t_amp), "freq": (k1, k2)})
            if len(B_specs) < nB:
                B_specs += _draw_specs(rng1, nB - len(B_specs))

        # coverage logs (local arc indices)
        from collections import Counter as _Counter
        def _counts(specs, key):
            idx = [int(s[key][0]) for s in specs if isinstance(s.get(key), (tuple, list))]
            return dict(_Counter(idx))

        print(f"[metrics] Class 0 — phi local counts:", _counts(A_specs, "phi"))
        print(f"[metrics] Class 0 — amp local counts:", _counts(A_specs, "amp"))
        print(f"[metrics] Class 1 — phi local counts:", _counts(B_specs, "phi"))
        print(f"[metrics] Class 1 — amp local counts:", _counts(B_specs, "amp"))

        # strict guarantees
        if len(A_specs) != int(nA) or len(B_specs) != int(nB):
            raise RuntimeError(f"[collect] got A={len(A_specs)}/{nA}, B={len(B_specs)}/{nB}")
        if any(not isinstance(s, dict) for s in A_specs + B_specs):
            bad = [i for i, s in enumerate(A_specs + B_specs) if not isinstance(s, dict)]
            raise RuntimeError(f"[collect] non-dict spec(s) at indices {bad[:5]}")

        return A_specs, B_specs

    def __len__(self): return len(self.images)
    def __getitem__(self, i):
        return torch.from_numpy(self.images[i])[None], torch.tensor(self.labels[i])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics & Plots
# ─────────────────────────────────────────────────────────────────────────────

def print_latent_distance_stats(dataset: "ShapeDeformDataset", normalize=True):

    # Banner: sampling/metrics mode (safe even if attrs are missing)
    try:
        mode = "INDEPENDENT" if bool(getattr(dataset, "independent_B", True)) else "PAIRED"
        seed = getattr(dataset, "metrics_seed", None)
        dbg  = getattr(dataset, "metrics_debug", None)
        print(f"[stats] Sampling mode for metrics: {mode}")
        print(f"[stats] metrics_seed={seed} metrics_debug={dbg}")
    except Exception:
        pass



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

    # Banner: sampling/metrics mode (safe even if attrs are missing)
    try:
        mode = "INDEPENDENT" if bool(getattr(dataset, "independent_B", True)) else "PAIRED"
        seed = getattr(dataset, "metrics_seed", None)
        dbg  = getattr(dataset, "metrics_debug", None)
        print(f"[stats] Sampling mode for metrics: {mode}")
        print(f"[stats] metrics_seed={seed} metrics_debug={dbg}")
    except Exception:
        pass


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
    """v13-style Phase–Amplitude Torus with GLOBAL cells.
    - Colors purely by class (0 vs 1), one class per cell.
    - Draws cell gridlines to make the checkerboard obvious.
    - Prints consistency checks.
    """
    Z = np.asarray(dataset.latents, dtype=np.float32)
    labels = np.asarray(dataset.labels, dtype=np.int64)

    # Recover angles (same ones used for latents)
    phi = (np.arctan2(Z[:,1], Z[:,0]) + 2*np.pi) % (2*np.pi)
    chi = (np.arctan2(Z[:,3], Z[:,2]) + 2*np.pi) % (2*np.pi)

    # Params
    try:
        m_phase = int(_CFG.get('m_phase', 6)); m_amp = int(_CFG.get('m_amp', 6)); gap = float(_CFG.get('gap', 0.15))
    except Exception:
        m_phase, m_amp, gap = 6, 6, 0.15

    # Global bands
    pg = np.array([global_band_index(t, m_phase, gap) for t in phi], dtype=np.int32)
    ag = np.array([global_band_index(t, m_amp,   gap) for t in chi], dtype=np.int32)

    # Consistency check: label should equal XOR of global indices
    lab_xor = ((pg + ag) & 1).astype(np.int64)
    mism = np.nonzero(lab_xor != labels)[0]
    print(f"[torus] checkerboard cells: {'pure' if len(mism)==0 else 'MISMATCH '+str(len(mism))}")

    # Plot cell-based to avoid visual mixing
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Draw gridlines at band boundaries
    base_w_phi = 2*np.pi / (2*m_phase)
    base_w_amp = 2*np.pi / (2*m_amp)
    for k in range(2*m_phase+1):
        x = k*base_w_phi
        ax.axvline(x, ls=':', lw=0.8, alpha=0.25, color='black')
    for j in range(2*m_amp+1):
        y = j*base_w_amp
        ax.axhline(y, ls=':', lw=0.8, alpha=0.25, color='black')

    # Colors and markers (high contrast to avoid confusion)
    C = {0: dict(c='tab:blue', marker='o'), 1: dict(c='tab:orange', marker='s')}

    # Group by GLOBAL cell to guarantee purity-by-construction on the figure
    for k in range(2*m_phase):
        for j in range(2*m_amp):
            idx = (pg==k) & (ag==j)
            if not idx.any(): 
                continue
            cl = int(labels[idx][0])
            ax.scatter(phi[idx], chi[idx], s=12, alpha=0.9, **C[cl])

    ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    ticklabs = ["0","π/2","π","3π/2","2π"]
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabs)
    ax.set_yticks(ticks); ax.set_yticklabels(ticklabs)
    h0 = plt.Line2D([0],[0], **C[0], lw=0, label="Class 0")
    h1 = plt.Line2D([0],[0], **C[1], lw=0, label="Class 1")
    ax.legend(handles=[h0,h1], loc="upper right")
    ax.set_xlabel("Phase angle φ"); ax.set_ylabel("Amplitude angle χ")
    ax.set_title("Phase–Amplitude Torus (GLOBAL cells, class-colored)")
    ax.grid(False)
    plt.tight_layout(); plt.show()
# --- v13 helper (minimal re-introduction) ---
def _phase_angles_from_latents(dataset):
    Z = np.asarray(dataset.latents, dtype=np.float32)
    phi1 = np.arctan2(Z[:, 1], Z[:, 0])
    phi2 = np.arctan2(Z[:, 3], Z[:, 2])  # kept for API parity; not used here
    phi1 = (phi1 + 2*np.pi) % (2*np.pi)
    phi2 = (phi2 + 2*np.pi) % (2*np.pi)
    return phi1, phi2

def plot_phase_ring_with_amp_bands(dataset, band_gap=0.15, alpha=0.85):
    """
    v13-style ring view:
      • Outer ring = ALL phase (φ) arcs
      • Inner ring = ALL amplitude (χ) arcs
      • Color by class (C0/C1). χ-parity shown by marker (even='o', odd='x').
    """
    import numpy as np, math, matplotlib.pyplot as plt
    Z = np.asarray(dataset.latents, dtype=np.float32)
    labs = np.asarray(dataset.labels, dtype=np.int64)
    phi = (np.arctan2(Z[:,1], Z[:,0]) + 2*np.pi) % (2*np.pi)
    chi = (np.arctan2(Z[:,3], Z[:,2]) + 2*np.pi) % (2*np.pi)

    # read params (fallbacks ok)
    try:
        gap   = float(_CFG.get("gap", band_gap))
        m_phi = int(_CFG.get("m_phase", 6))
        m_amp = int(_CFG.get("m_amp", 6))
    except Exception:
        gap, m_phi, m_amp = band_gap, 6, 6

    def global_band_index(theta, m, gap_frac):
        theta = float(theta) % (2*math.pi)
        base  = 2*math.pi/(2*m)
        return int(math.floor((theta - gap_frac*base/2)/base)) % (2*m)

    # bands & χ-parity
    ag = np.array([global_band_index(t, m_amp, gap) for t in chi], dtype=np.int32)
    chi_parity = ag & 1

    # coords
    Xφ = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    Xχ = np.stack([np.cos(chi), np.sin(chi)], axis=1)
    r_outer = 1.0
    r_inner = max(0.0, 1.0 - band_gap)

    fig, ax = plt.subplots(figsize=(6.2,6.2))
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t)*r_outer, np.sin(t)*r_outer, lw=1.0, color="black", alpha=0.6)
    ax.plot(np.cos(t)*r_inner, np.sin(t)*r_inner, lw=1.0, color="black", alpha=0.35)

    # faint band strokes
    def draw_band_grid(m, r, a=0.10):
        base = 2*np.pi/(2*m)
        for k in range(2*m):
            s = k*base + gap*base/2
            e = (k+1)*base - gap*base/2
            tt = np.linspace(s, e, 64)
            ax.plot(np.cos(tt)*r, np.sin(tt)*r, lw=3.0, alpha=a, color="grey")
    draw_band_grid(m_phi, r_outer, a=0.10)
    draw_band_grid(m_amp, r_inner, a=0.08)

    # colors and markers (v13 look)
    c0, c1 = "green", "lime"
    even = (chi_parity==0); odd = ~even

    # outer φ ring (all φ samples by class)
    ax.scatter(Xφ[labs==0,0]*r_outer, Xφ[labs==0,1]*r_outer, s=14, alpha=alpha, color=c0, marker='o', label="C0 φ")
    ax.scatter(Xφ[labs==1,0]*r_outer, Xφ[labs==1,1]*r_outer, s=14, alpha=alpha, color=c1, marker='o', label="C1 φ")

    # inner χ ring (all χ samples; χ-parity shown by marker)
    ax.scatter(Xχ[(even)&(labs==0),0]*r_inner, Xχ[(even)&(labs==0),1]*r_inner, s=16, alpha=alpha, color=c0, marker='o', label="C0 even χ-arc")
    ax.scatter(Xχ[(odd )&(labs==0),0]*r_inner, Xχ[(odd )&(labs==0),1]*r_inner, s=18, alpha=alpha, color=c0, marker='x', label="C0 odd χ-arc")
    ax.scatter(Xχ[(even)&(labs==1),0]*r_inner, Xχ[(even)&(labs==1),1]*r_inner, s=16, alpha=alpha, color=c1, marker='o', label="C1 even χ-arc")
    ax.scatter(Xχ[(odd )&(labs==1),0]*r_inner, Xχ[(odd )&(labs==1),1]*r_inner, s=18, alpha=alpha, color=c1, marker='x', label="C1 odd χ-arc")

    # dedup legend
    H,L = ax.get_legend_handles_labels()
    seen, H2, L2 = set(), [], []
    for h,l in zip(H,L):
        if l not in seen: seen.add(l); H2.append(h); L2.append(l)
    ax.legend(H2, L2, loc="upper right")

    ax.set_aspect("equal","box")
    ax.set_title("Phase Ring with Amplitude Parity Bands (via χ)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.25)
    plt.tight_layout(); plt.show()


def plot_phase_ring_with_amp_bands(dataset, band_gap=0.15, alpha=0.85):
    """
    v13-style ring view:
      • Outer ring = ALL phase (φ) arcs
      • Inner ring = ALL amplitude (χ) arcs
      • Color by class (C0/C1). χ-parity shown by marker (even='o', odd='x').
    """
    import numpy as np, math, matplotlib.pyplot as plt
    Z = np.asarray(dataset.latents, dtype=np.float32)
    labs = np.asarray(dataset.labels, dtype=np.int64)
    phi = (np.arctan2(Z[:,1], Z[:,0]) + 2*np.pi) % (2*np.pi)
    chi = (np.arctan2(Z[:,3], Z[:,2]) + 2*np.pi) % (2*np.pi)

    # read params (fallbacks ok)
    try:
        gap   = float(_CFG.get("gap", band_gap))
        m_phi = int(_CFG.get("m_phase", 6))
        m_amp = int(_CFG.get("m_amp", 6))
    except Exception:
        gap, m_phi, m_amp = band_gap, 6, 6

    def global_band_index(theta, m, gap_frac):
        theta = float(theta) % (2*math.pi)
        base  = 2*math.pi/(2*m)
        return int(math.floor((theta - gap_frac*base/2)/base)) % (2*m)

    # bands & χ-parity
    ag = np.array([global_band_index(t, m_amp, gap) for t in chi], dtype=np.int32)
    chi_parity = ag & 1

    # coords
    Xφ = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    Xχ = np.stack([np.cos(chi), np.sin(chi)], axis=1)
    r_outer = 1.0
    r_inner = max(0.0, 1.0 - band_gap)

    fig, ax = plt.subplots(figsize=(6.2,6.2))
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t)*r_outer, np.sin(t)*r_outer, lw=1.0, color="black", alpha=0.6)
    ax.plot(np.cos(t)*r_inner, np.sin(t)*r_inner, lw=1.0, color="black", alpha=0.35)

    # faint band strokes
    def draw_band_grid(m, r, a=0.10):
        base = 2*np.pi/(2*m)
        for k in range(2*m):
            s = k*base + gap*base/2
            e = (k+1)*base - gap*base/2
            tt = np.linspace(s, e, 64)
            ax.plot(np.cos(tt)*r, np.sin(tt)*r, lw=3.0, alpha=a, color="grey")
    draw_band_grid(m_phi, r_outer, a=0.10)
    draw_band_grid(m_amp, r_inner, a=0.08)

    # colors and markers (v13 look)
    c0, c1 = "green", "lime"
    even = (chi_parity==0); odd = ~even

    # outer φ ring (all φ samples by class)
    ax.scatter(Xφ[labs==0,0]*r_outer, Xφ[labs==0,1]*r_outer, s=14, alpha=alpha, color=c0, marker='o', label="C0 φ")
    ax.scatter(Xφ[labs==1,0]*r_outer, Xφ[labs==1,1]*r_outer, s=14, alpha=alpha, color=c1, marker='o', label="C1 φ")

    # inner χ ring (all χ samples; χ-parity shown by marker)
    ax.scatter(Xχ[(even)&(labs==0),0]*r_inner, Xχ[(even)&(labs==0),1]*r_inner, s=16, alpha=alpha, color=c0, marker='o', label="C0 even χ-arc")
    ax.scatter(Xχ[(odd )&(labs==0),0]*r_inner, Xχ[(odd )&(labs==0),1]*r_inner, s=18, alpha=alpha, color=c0, marker='x', label="C0 odd χ-arc")
    ax.scatter(Xχ[(even)&(labs==1),0]*r_inner, Xχ[(even)&(labs==1),1]*r_inner, s=16, alpha=alpha, color=c1, marker='o', label="C1 even χ-arc")
    ax.scatter(Xχ[(odd )&(labs==1),0]*r_inner, Xχ[(odd )&(labs==1),1]*r_inner, s=18, alpha=alpha, color=c1, marker='x', label="C1 odd χ-arc")

    # dedup legend
    H,L = ax.get_legend_handles_labels()
    seen, H2, L2 = set(), [], []
    for h,l in zip(H,L):
        if l not in seen: seen.add(l); H2.append(h); L2.append(l)
    ax.legend(H2, L2, loc="upper right")

    ax.set_aspect("equal","box")
    ax.set_title("Phase Ring with Amplitude Parity Bands (via χ)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, ls="--", alpha=0.25)
    plt.tight_layout(); plt.show()

def plot_phase_ring_given_amp_band(dataset, amp_band, band_gap=0.15):
    """
    Show the φ (phase) ring but ONLY for samples in χ global band = amp_band.
    Result: each visible φ-arc is pure-class and alternates C0/C1 around the ring.
    """
    import numpy as np, math, matplotlib.pyplot as plt

    Z = np.asarray(dataset.latents, dtype=np.float32)
    labs = np.asarray(dataset.labels, dtype=np.int64)
    phi = (np.arctan2(Z[:,1], Z[:,0]) + 2*np.pi) % (2*np.pi)
    chi = (np.arctan2(Z[:,3], Z[:,2]) + 2*np.pi) % (2*np.pi)

    try:
        m_phase = int(_CFG.get('m_phase', 6)); m_amp = int(_CFG.get('m_amp', 6)); gap=float(_CFG.get('gap', band_gap))
    except Exception:
        m_phase, m_amp, gap = 6, 6, band_gap

    def global_band_index(theta, m, gap_frac):
        base_w = 2*np.pi/(2*m); return int(math.floor(((theta % (2*np.pi)) - gap_frac*base_w/2)/base_w))%(2*m)

    ag = np.array([global_band_index(t, m_amp, gap) for t in chi], dtype=np.int32)
    sel = (ag == int(amp_band))
    if not sel.any():
        print(f"[phase-ring] No samples in χ-band {amp_band}")
        return

    X = np.stack([np.cos(phi[sel]), np.sin(phi[sel])], axis=1)
    r = 1.0
    t = np.linspace(0, 2*np.pi, 512)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(np.cos(t)*r, np.sin(t)*r, lw=1, color="black", alpha=0.6)
    ax.scatter(X[labs[sel]==0,0]*r, X[labs[sel]==0,1]*r, s=16, alpha=0.9, color="green", marker='o', label="C0")
    ax.scatter(X[labs[sel]==1,0]*r, X[labs[sel]==1,1]*r, s=16, alpha=0.9, color="lime",  marker='o', label="C1")
    ax.legend(loc="upper right"); ax.set_aspect("equal","box")
    ax.set_title(f"Phase ring (χ-band fixed = {amp_band}) — arcs are pure-class")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True, ls="--", alpha=0.25)
    plt.tight_layout(); plt.show()


def plot_amp_ring_given_phase_band(dataset, phase_band, band_gap=0.15):
    """
    Show the χ (amplitude) ring but ONLY for samples in φ global band = phase_band.
    Result: each visible χ-arc is pure-class and alternates C0/C1 around the ring.
    """
    import numpy as np, math, matplotlib.pyplot as plt

    Z = np.asarray(dataset.latents, dtype=np.float32)
    labs = np.asarray(dataset.labels, dtype=np.int64)
    phi = (np.arctan2(Z[:,1], Z[:,0]) + 2*np.pi) % (2*np.pi)
    chi = (np.arctan2(Z[:,3], Z[:,2]) + 2*np.pi) % (2*np.pi)

    try:
        m_phase = int(_CFG.get('m_phase', 6)); m_amp = int(_CFG.get('m_amp', 6)); gap=float(_CFG.get('gap', band_gap))
    except Exception:
        m_phase, m_amp, gap = 6, 6, band_gap

    def global_band_index(theta, m, gap_frac):
        base_w = 2*np.pi/(2*m); return int(math.floor(((theta % (2*np.pi)) - gap_frac*base_w/2)/base_w))%(2*m)

    pg = np.array([global_band_index(t, m_phase, gap) for t in phi], dtype=np.int32)
    sel = (pg == int(phase_band))
    if not sel.any():
        print(f"[amp-ring] No samples in φ-band {phase_band}")
        return

    X = np.stack([np.cos(chi[sel]), np.sin(chi[sel])], axis=1)
    r_outer = 1.0; r = max(0.0, 1.0 - band_gap)
    t = np.linspace(0, 2*np.pi, 512)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(np.cos(t)*r, np.sin(t)*r, lw=1, color="black", alpha=0.6)
    ax.scatter(X[labs[sel]==0,0]*r, X[labs[sel]==0,1]*r, s=16, alpha=0.9, color="green", marker='o', label="C0")
    ax.scatter(X[labs[sel]==1,0,]*r, X[labs[sel]==1,1]*r, s=16, alpha=0.9, color="lime",  marker='o', label="C1")
    ax.legend(loc="upper right"); ax.set_aspect("equal","box")
    ax.set_title(f"Amplitude ring (φ-band fixed = {phase_band}) — arcs are pure-class")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(True, ls="--", alpha=0.25)
    plt.tight_layout(); plt.show()


def show_examples_with_nearest(dataset: "ShapeDeformDataset", k: int = 8):
    """
    Top row: k exemplars from Class 0.
    Bottom row: for each top image, the *Studio-boundary* nearest shape from Class 1.
    """
    imgs = dataset.images
    labs = dataset.labels
    A_idx = np.where(labs == 0)[0]
    B_idx = np.where(labs == 1)[0]
    if len(A_idx) == 0 or len(B_idx) == 0:
        raise RuntimeError("Both classes must be present for nearest visualization.")

    k = int(min(k, len(A_idx), len(B_idx)))
    A_idx = A_idx[:k]

    # Pre-render Class-1 pool boundaries once (faster)
    B_bounds = []
    for j in B_idx:
        spec_j = dataset.specs[j]
        with _SpecApply(_STUDIO, spec_j):
            xB, yB, *_ = _STUDIO._render_with_spec(1, spec_j)
        B_bounds.append((xB, yB))

    nn_for_A = []
    for i in A_idx:
        spec_i = dataset.specs[i]
        # Render the source boundary (Class 0) once
        with _SpecApply(_STUDIO, spec_i):
            xA, yA, *_ = _STUDIO._render_with_spec(0, spec_i)

        # Boundary-distance argmin over B pool
        best_j = 0
        best_d = float("inf")
        for jj, (xB, yB) in enumerate(B_bounds):
            d = _STUDIO._distance_xy(xA, yA, xB, yB)
            if d < best_d:
                best_d, best_j = d, jj
        nn_for_A.append(B_idx[best_j])

    # Plot (same layout as before)
    fig, axes = plt.subplots(2, k, figsize=(1.6*k, 3.4))
    for j in range(k):
        axes[0, j].imshow(imgs[A_idx[j]], cmap="gray", vmin=0, vmax=1)
        axes[0, j].axis("off"); axes[0, j].set_title("C0")
        axes[1, j].imshow(imgs[nn_for_A[j]], cmap="gray", vmin=0, vmax=1)
        axes[1, j].axis("off"); axes[1, j].set_title("C1⋆")
    plt.suptitle("Exemplars (top) & Closest Matches by Studio Boundary (bottom)")
    plt.tight_layout()
    plt.show()


def _save_gray_png(path: str, img: np.ndarray):
    from PIL import Image
    arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def export_xab_and_categorisation(dataset, out_root: Path, excel_out: Path,
                                  n_within=100, n_between=100, seed=123):
    """
    Create 200 pairs (100 within, 100 between), save images under ./experimentFiles/gabors/testing/pair{i}/,
    and write xab.xlsx and categorisation.xlsx in excel_out. Paths in Excel are relative to project root ("./experimentFiles/...").

    Unchanged logic, but 'nearest' is computed with Studio boundary distance.
    """
    import numpy as _np
    from pathlib import Path as _Path
    from PIL import Image as _Image
    import pandas as _pd

    rng  = _np.random.default_rng(seed)
    imgs = _np.asarray(dataset.images)
    labs = _np.asarray(dataset.labels).astype(int)

    idx0 = _np.where(labs == 0)[0]
    idx1 = _np.where(labs == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        raise RuntimeError("Both classes must be present.")

    # --- helpers (Studio boundary) -------------------------------------------------
    def _render_boundary(cls_id: int, spec: dict):
        with _SpecApply(_STUDIO, spec):
            x, y, *_ = _STUDIO._render_with_spec(cls_id, spec)
        return x, y

    def _nearest_by_boundary(a_idx: int, pool_indices, cls_src: int, cls_tgt: int):
        """Return dataset index in pool_indices with minimum Studio boundary distance to a_idx."""
        spec_a = dataset.specs[a_idx]
        xA, yA = _render_boundary(cls_src, spec_a)
        # scan pool
        best_i, best_d = None, float("inf")
        for p in pool_indices:
            xp, yp = _render_boundary(cls_tgt, dataset.specs[int(p)])
            d = _STUDIO._distance_xy(xA, yA, xp, yp)
            if d < best_d:
                best_d, best_i = d, int(p)
        return best_i
    # -----------------------------------------------------------------------------

    # WITHIN pairs (unchanged structure)
    within = []
    choose = rng.integers(0, 2, size=n_within)
    for t in range(n_within):
        if choose[t] == 0:
            i = int(rng.integers(0, len(idx0))); a = idx0[i]
            b = _nearest_by_boundary(a, idx0, cls_src=0, cls_tgt=0)
            if b == a and len(idx0) > 1:
                pool = [p for p in idx0 if p != a]
                b = _nearest_by_boundary(a, pool, cls_src=0, cls_tgt=0)
            within.append((a, b))
        else:
            i = int(rng.integers(0, len(idx1))); a = idx1[i]
            b = _nearest_by_boundary(a, idx1, cls_src=1, cls_tgt=1)
            if b == a and len(idx1) > 1:
                pool = [p for p in idx1 if p != a]
                b = _nearest_by_boundary(a, pool, cls_src=1, cls_tgt=1)
            within.append((a, b))

    # BETWEEN pairs (unchanged structure)
    between = []
    choose = rng.integers(0, 2, size=n_between)
    for t in range(n_between):
        if choose[t] == 0:
            i = int(rng.integers(0, len(idx0))); a = idx0[i]
            b = _nearest_by_boundary(a, idx1, cls_src=0, cls_tgt=1)
            between.append((a, b))
        else:
            i = int(rng.integers(0, len(idx1))); a = idx1[i]
            b = _nearest_by_boundary(a, idx0, cls_src=1, cls_tgt=0)
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

    # --- Excel outputs (unchanged)
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
    k_max=8, m_freq=6, gap_freq=0.50,k_min1=3,k_min2=3,

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
    tinys: List[Tuple[int,float,float,float]] = None,
    parity_flip: bool=False
):
    # Capture JSON as the single source of truth for dataset sampling
    # _CFG = {
    #     'm_phase': m_phase, 'm_amp': m_amp, 'gap': gap, 'phase_deg': phase_deg,
    #     'which_arc_phi': which_arc_phi, 'pos_phi': pos_phi,
    #     'which_arc_amp': which_arc_amp, 'pos_amp': pos_amp,
    #     'R': R, 'profile': profile, 'sharp': sharp, 'amp_min': amp_min, 'amp_max': amp_max,
    #     'k_max': k_max, 'm_freq': m_freq, 'gap_freq': gap_freq,
    #     'k_min1': k_min1, 'k_min2': k_min2,
    #     'phase_src': phase_src, 'amp_src': amp_src, 'freq_src': freq_src,
    #     'k1': k1, 'a1': a1, 'phi1_deg': phi1_deg,
    #     'phase_mode1': phase_mode1, 'sphi1': sphi1, 'Kphi1': Kphi1,
    #     'amp_mode1': amp_mode1, 'sA1': sA1, 'KA1': KA1,
    #     'k2': k2, 'a2': a2, 'phi2_deg': phi2_deg,
    #     'phase_mode2': phase_mode2, 'sphi2': sphi2, 'Kphi2': Kphi2,
    #     'amp_mode2': amp_mode2, 'sA2': sA2, 'KA2': KA2,
    #     'tinys': tinys, 'parity_flip': parity_flip,
    # }

    # Global rings
    _STUDIO.m_phase=int(m_phase); _STUDIO.m_amp=int(m_amp)
    _STUDIO.gap=float(gap); _STUDIO.phase_deg=float(phase_deg)
    _STUDIO.which_arc_phi=int(which_arc_phi); _STUDIO.pos_phi=float(pos_phi)
    _STUDIO.which_arc_amp=int(which_arc_amp); _STUDIO.pos_amp=float(pos_amp)
    _STUDIO.R=float(R); _STUDIO.profile=str(profile); _STUDIO.sharp=float(sharp)
    if amp_min is not None: _STUDIO.amp_min=float(amp_min)
    if amp_max is not None: _STUDIO.amp_max=float(amp_max)


    try:
        _STUDIO.s_amp_min.set_value(_STUDIO.amp_min)
        _STUDIO.s_amp_max.set_value(_STUDIO.amp_max)
    except Exception:
        pass

    # Frequency ring
    _STUDIO.k_max=int(k_max); _STUDIO.m_freq=int(m_freq); _STUDIO.gap_freq=float(gap_freq)

    # Sources (robust selection: text first, then index fallback if it contains "none")
    _STUDIO.cmb_phase_src.setCurrentText(str(phase_src))
    _STUDIO.cmb_amp_src.setCurrentText(str(amp_src))
    _STUDIO.cmb_freq_src.setCurrentText(str(freq_src))

    # Frequency minima (Studio state + UI if available)
    _STUDIO.kmin1 = int(k_min1)
    _STUDIO.kmin2 = int(k_min2)

    # Keep k_max ≥ both minima
    _STUDIO.k_max = max(int(_STUDIO.k_max), _STUDIO.kmin1, _STUDIO.kmin2)

    # If Studio has sliders, sync them (optional but nice)
    try:
        _STUDIO.s_kmin1.set_value(_STUDIO.kmin1)
        _STUDIO.s_kmin2.set_value(_STUDIO.kmin2)
        _STUDIO.s_kmax.slider.setMinimum(int(max(_STUDIO.kmin1, _STUDIO.kmin2)))
        # also ensure visible value respects the new floor
        _STUDIO.s_kmax.set_value(int(max(_STUDIO.s_kmax.value(), _STUDIO.kmin1, _STUDIO.kmin2)))
    except Exception:
        pass

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
    nA = 100   # class 0
    nB = nA   # class 1

    # Image/output settings
    image_size = 128
    intensity = 1.0
    bg = 0.0
    norm = "max"   # "max" | "l2" | "none"
    seed = 42
    batch_size = 64
    R=5.0
    profile="absolute"
    sharp=0.30

    m_phase = 6
    m_amp = 6
    m_freq = 6

    amp_min=2
    amp_max=3
    k_min1 = 2
    k_min2 = 2
    k_max = 6



    # Actions
    PRINT_STATS = False
    SHOW_EXAMPLES = True
    export_gabors = False

    # Example: drive Studio from a saved Studio JSON
    settings_path = "./studio_settings.json"  # change to your saved file
    cfg = load_studio_settings_json(settings_path)




    # Ensure new freq minima are supported by the signature (add k_min1/k_min2 there if you haven't yet)
    set_studio_from_main(**cfg)  # one-liner, names match exactly

    # Studio preference (all sliders/knobs exposed exactly like older mains)
    # set_studio_from_main(
    #     m_phase=m_phase, m_amp=m_amp, gap=0.1, phase_deg=0.0,
    #     which_arc_phi=0, pos_phi=0.5, which_arc_amp=0, pos_amp=0.5,
    #     R=R, profile=profile, sharp=sharp, amp_min=amp_min, amp_max=amp_max,   # for example
    #     k_max=k_max, m_freq=m_freq, gap_freq=0.1,k_min1=k_min1,k_min2=k_min2,
    #     phase_src="None (ring)", amp_src="None (ring)", freq_src="None (ring)",
    #     k1=3, a1=0.9, phi1_deg=45.0, phase_mode1="signed_absolute", sphi1=0.25, Kphi1=10.0,
    #     amp_mode1="relative", sA1=0.25, KA1=10.0,
    #     k2=5, a2=0.6, phi2_deg=0.0, phase_mode2="signed_absolute", sphi2=0.25, Kphi2=10.0,
    #     amp_mode2="relative", sA2=0.25, KA2=10.0,
    #     tinys=[(5, 1.0, 0.0, 1.0), (14, 0.0, 0.0, 0.8), (2, 0.0, 0.0, 0.3),
    #     parity_flip: bool=False]
    # )
    # Ensure Studio has a valid, non-zero base state before specs are collected
    #_STUDIO.resample_exemplars()

    # Build dataset by asking Studio to generate batches until we have nA/nB
    ds = ShapeDeformDataset(nA=nA, nB=nB,
                            image_size=image_size,
                            intensity=intensity, bg=bg, norm=norm,
                            seed=seed, batch=500,independent_B=True,
                            metrics_seed=12345,
                            metrics_debug=True)


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

    plot_phase_ring_with_amp_bands(ds, band_gap=0.15)
    plot_phase_ring_given_amp_band(ds, amp_band=0)
    plot_amp_ring_given_phase_band(ds, phase_band=0)
    plot_phase_amp_torus_v13(ds)

if __name__ == "__main__":
    main()
