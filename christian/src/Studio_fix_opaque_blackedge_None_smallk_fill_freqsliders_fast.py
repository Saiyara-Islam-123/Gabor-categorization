#!/usr/bin/env python3
# ShapeStudio (PySide6 + PyQtGraph) — dual-class live preview + exemplar pairing with None/Fixed sources
# Deps: pip install PySide6 pyqtgraph numpy
import math
import sys
from dataclasses import dataclass
import numpy as np

from PySide6 import QtGui
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

try:
    from PySide6 import QtCore, QtWidgets
except ImportError:
    from PyQt5 import QtCore, QtWidgets



TAU = math.tau if hasattr(math, "tau") else 2*math.pi

def wrap_signed(x: float) -> float:
    return ((x + math.pi) % TAU + TAU) % TAU - math.pi

def ring_phase_for_class(cls: int, m_arcs: int, gap_frac: float, phase_deg: float, which_arc: int, pos_in_arc: float,PARITY_FLIP) -> float:
    total = max(1, 2*int(m_arcs))
    base_sector = TAU / total
    usable = (1.0 - gap_frac) * base_sector
    off = math.radians(phase_deg)
    par = (1 if cls == 1 else 0)
    if PARITY_FLIP:
        par ^= 1  # flip 0<->1
    global_sector = (int(which_arc) * 2 + par) % total  # in ring_phase_for_class
    start = global_sector * base_sector + off + 0.5*gap_frac*base_sector
    pos_in_arc = max(0.0, min(1.0, pos_in_arc))
    phi = start + pos_in_arc * usable
    return (phi % TAU + TAU) % TAU

def amp_from_arc(
    cls: int, m_arcs: int, gap_frac: float,
    amp_min: float, amp_max: float,
    which_arc: int, pos_in_arc: float, PARITY_FLIP
) -> float:
    # total bands around the ring
    total = max(1, 2 * int(m_arcs))
    usable = (1.0 - gap_frac)
    band_width = usable / total

    # ---- key change: decouple amplitude from class parity ----
    # use the high/low half of pos_in_arc to choose band parity (0 or 1)
    t = max(0.0, min(1.0, float(pos_in_arc)))
    if t < 0.5:
        par = 0
        t_local = t * 2.0       # map [0,0.5) -> [0,1) inside the chosen band
    else:
        par = 1
        t_local = (t - 0.5) * 2.0

    # global band index: depends on which_arc and the *derived* parity, NOT on cls
    global_band = (int(which_arc) * 2 + par) % total

    # compute amplitude interval for that band
    lo = amp_min + (global_band * band_width + gap_frac / 2.0) * (amp_max - amp_min)
    hi = lo + band_width * (amp_max - amp_min)

    return max(amp_min, min(amp_max, lo + t_local * (hi - lo)))


def apply_phase_offset(phi_ring: float, phi_base: float, mode: str, strength: float, divisor: float) -> float:
    if phi_base is None:
        return phi_ring
    if mode == "signed_absolute":
        return (phi_base + wrap_signed(phi_ring) / max(divisor or 0.0, 1e-9)) % TAU
    d = wrap_signed(phi_ring - phi_base)
    return (phi_base + float(strength or 0.0) * d) % TAU

def apply_amp_offset(a_ring: float, a_base: float, mode: str, strength: float, divisor: float) -> float:
    if a_base is None:
        return a_ring
    if mode == "signed_absolute":
        return float(a_base) + float(a_ring) / max(divisor or 0.0, 1e-9)
    return (1.0 - float(strength or 0.0)) * float(a_base) + float(strength or 0.0) * float(a_ring)

@dataclass
class BaseHarm:
    k: int
    a: float
    phi: float
    pmode: str
    pstr: float
    pdiv: float
    amode: str
    astr: float
    adiv: float

@dataclass
class TinyHarm:
    k: int
    a: float
    phi: float
    weight: float

class ControlRow(QtWidgets.QWidget):
    changed = QtCore.Signal()
    def __init__(self, label: str, vmin: float, vmax: float, v0: float, step: float = 1.0, is_int=False, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.lab = QtWidgets.QLabel(label); self.val = QtWidgets.QLabel("")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.is_int = is_int
        self.scale = 1 if is_int else int(round(1/step))
        self.slider.setMinimum(int(round(vmin*self.scale)))
        self.slider.setMaximum(int(round(vmax*self.scale)))
        self.slider.setValue(int(round(v0*self.scale)))
        self.slider.valueChanged.connect(self._emit)
        lay.addWidget(self.lab, 2); lay.addWidget(self.slider, 10); lay.addWidget(self.val, 2)
        self.set_value(v0)
    def value(self):
        v = self.slider.value()/self.scale
        return int(round(v)) if self.is_int else v
    def set_value(self, v):
        self.slider.setValue(int(round(v*self.scale))); self._emit(self.slider.value())
    def _emit(self, _):
        v = self.value()
        self.val.setText(str(v if self.is_int else f"{v:.2f}"))
        self.changed.emit()

class ShapeStudio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShapeStudio — None/Fixed sources + live nearest exemplars")
        self.resize(1320, 860)

        # ----- state (kept) -----
        self.m_phase = 2
        self.m_amp = 2
        self.gap = 0.25
        self.phase_deg = 0.0
        self.which_arc_phi = 0; self.pos_phi = 0.5
        self.which_arc_amp = 0; self.pos_amp = 0.5
        self.R = 5.0
        self.profile = "absolute"
        self.sharp = 0.3
        self.amp_min = 0.5; self.amp_max = 3.5
        # Frequency sampling state (None mode)
        self.k_max = 6
        self.m_freq = 6
        self.kmin1 = 2
        self.kmin2 = 3

        self.gap_freq = 0.30
        self.bases = [
            BaseHarm(k=3, a=0.9, phi=math.radians(45), pmode="signed_absolute", pstr=0.25, pdiv=10.0,
                     amode="relative", astr=0.25, adiv=10.0),
            BaseHarm(k=5, a=0.6, phi=0.0, pmode="signed_absolute", pstr=0.25, pdiv=10.0,
                     amode="relative", astr=0.25, adiv=10.0),
        ]
        self.tinys = [
            TinyHarm(k=6, a=0.0, phi=0.0, weight=0.5),
            TinyHarm(k=14, a=0.0, phi=0.0, weight=0.8),
            TinyHarm(k=2, a=0.0, phi=0.0, weight=0.3),
        ]

        self.N = 1600
        self.thetas = np.linspace(0, TAU, self.N, endpoint=False)

        self._stats_timer = QtCore.QTimer(self)
        self._stats_timer.setSingleShot(True)
        self._stats_timer.setInterval(250)  # ms
        self._stats_timer.timeout.connect(self._update_live_stats)

        # ----- UI -----
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left controls
        ctrlScroll = QtWidgets.QScrollArea(); ctrlScroll.setWidgetResizable(True)
        ctrl = QtWidgets.QWidget(); ctrlScroll.setWidget(ctrl)
        controls = QtWidgets.QVBoxLayout(ctrl)

        ctrl.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        ctrl.setMinimumWidth(200)  # adjust to taste
        ctrl.setMaximumWidth(1500)

        def add_section(title):
            lab = QtWidgets.QLabel(title); f = lab.font(); f.setBold(True); lab.setFont(f)
            controls.addWidget(lab)

        # Ring / Dataset
        add_section("Ring / Dataset")
        self.s_m_phase = ControlRow("m_phase", 1, 10, self.m_phase, is_int=True); self.s_m_phase.changed.connect(self.update_from_controls)
        self.s_m_amp   = ControlRow("m_amp",   1, 10, self.m_amp,   is_int=True); self.s_m_amp.changed.connect(self.update_from_controls)
        self.s_gap     = ControlRow("gap",     0.0, 0.8, self.gap, 0.01);         self.s_gap.changed.connect(self.update_from_controls)
        self.s_phase   = ControlRow("phase°", -180, 180, self.phase_deg, 1.0);    self.changed_connect = self.s_phase.changed.connect(self.update_from_controls)
        controls.addWidget(self.s_m_phase); controls.addWidget(self.s_m_amp); controls.addWidget(self.s_gap); controls.addWidget(self.s_phase)

        # Frequency sampling controls
        add_section("Frequency sampling")
        self.s_kmin1 = ControlRow("k_min1", 1, 40, 2, is_int=True);
        self.s_kmin1.changed.connect(self.update_from_controls)
        self.s_kmin2 = ControlRow("k_min2", 1, 40, 3, is_int=True);
        self.s_kmin2.changed.connect(self.update_from_controls)
        self.s_kmax = ControlRow("k_max (freq None)", 4, 40, 8, is_int=True);
        self.s_kmax.changed.connect(self.update_from_controls)
        self.s_m_freq = ControlRow("m_freq (arcs)", 1, 10, 2, is_int=True);
        self.s_m_freq.changed.connect(self.update_from_controls)
        self.s_gap_freq = ControlRow("gap_freq", 0.0, 0.8, 0.30, 0.01);
        self.s_gap_freq.changed.connect(self.update_from_controls)
        controls.addWidget(self.s_kmin1);
        controls.addWidget(self.s_kmin2)
        controls.addWidget(self.s_kmax);
        controls.addWidget(self.s_m_freq);
        controls.addWidget(self.s_gap_freq)

        # Source selection (Fixed vs None)
        add_section("Source selection (Fixed vs None)")
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Phase source"))
        self.cmb_phase_src = QtWidgets.QComboBox(); self.cmb_phase_src.addItems(["Fixed", "None (ring)"]); self.cmb_phase_src.currentIndexChanged.connect(self.on_source_change)
        row.addWidget(self.cmb_phase_src); controls.addLayout(row)

        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Amplitude source"))
        self.cmb_amp_src = QtWidgets.QComboBox(); self.cmb_amp_src.addItems(["Fixed", "None (ring)"]); self.cmb_amp_src.currentIndexChanged.connect(self.on_source_change)
        row.addWidget(self.cmb_amp_src); controls.addLayout(row)

        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("Frequency source"))
        self.cmb_freq_src = QtWidgets.QComboBox(); self.cmb_freq_src.addItems(["Fixed", "None (ring)"]); self.cmb_freq_src.currentIndexChanged.connect(self.on_source_change)
        row.addWidget(self.cmb_freq_src); controls.addLayout(row)

        # Arc selection
        add_section("Arc selection")
        self.s_arc_phi = ControlRow("arc φ idx", 0, 9, self.which_arc_phi, is_int=True); self.s_arc_phi.changed.connect(self.update_from_controls)
        self.s_pos_phi = ControlRow("pos φ",     0.0, 1.0, self.pos_phi, 0.01);          self.s_pos_phi.changed.connect(self.update_from_controls)
        self.s_arc_amp = ControlRow("arc a idx", 0, 9, self.which_arc_amp, is_int=True); self.s_arc_amp.changed.connect(self.update_from_controls)
        self.s_pos_amp = ControlRow("pos a",     0.0, 1.0, self.pos_amp, 0.01);          self.s_pos_amp.changed.connect(self.update_from_controls)
        for w in (self.s_arc_phi, self.s_pos_phi, self.s_arc_amp, self.s_pos_amp): controls.addWidget(w)

        # Global & Sharp
        add_section("Global & Sharp")
        self.s_R     = ControlRow("R (px)", 1.0, 200.0, self.R, 1.0);          self.s_R.changed.connect(self.update_from_controls)
        self.profile_combo = QtWidgets.QComboBox(); self.profile_combo.addItems(["absolute","normalized"]); self.profile_combo.setCurrentText(self.profile)
        self.profile_combo.currentTextChanged.connect(self.update_from_controls)
        row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel("profile")); row.addWidget(self.profile_combo); row.addStretch(1)
        self.s_sharp = ControlRow("difficulty_sharp", 0.0, 1.0, self.sharp, 0.01); self.s_sharp.changed.connect(self.update_from_controls)
        controls.addWidget(self.s_R); cw = QtWidgets.QWidget(); cw.setLayout(row); controls.addWidget(cw); controls.addWidget(self.s_sharp)
        self.chk_hide_top = QtWidgets.QCheckBox("Hide top plots"); controls.addWidget(self.chk_hide_top)
        # Amplitude bounds (absolute units, same space as R)
        self.s_amp_min = ControlRow("amp_min", 0.0, 400.0, self.amp_min, 0.5);
        self.s_amp_min.changed.connect(self.update_from_controls)
        self.s_amp_max = ControlRow("amp_max", 0.0, 400.0, self.amp_max, 0.5);
        self.s_amp_max.changed.connect(self.update_from_controls)
        controls.addWidget(self.s_amp_min)
        controls.addWidget(self.s_amp_max)

        self.PARITY_FLIP = False  # set True to swap arc parity between classes

        # In __init__, after building the controls UI:
        self.btn_save = QtWidgets.QPushButton("Save Settings…")
        self.btn_save.clicked.connect(self.save_settings_to_file)
        controls.addWidget(self.btn_save)

        #btn_load = QtWidgets.QPushButton("Load settings (JSON)")
        #btn_load.clicked.connect(self.load_settings_from_file)
        #controls.addWidget(btn_load)

        # === Live distances (auto, no button) ===
        # self.lbl_stats_title = QtWidgets.QLabel("Live distances (dataset printers)")
        # self.lbl_stats_title.setAlignment(QtCore.Qt.AlignCenter)
        # self.lbl_stats_title.setStyleSheet("QLabel { font-size: 16pt; font-weight: 700; }")
        # controls.addWidget(self.lbl_stats_title)

        self.lbl_live = QtWidgets.QLabel("— Distances —")
        self.lbl_live.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_live.setStyleSheet("""
            QLabel {
                font-size: 10pt; font-weight: 800; padding: 8px 12px;
                border-radius: 12px; background: #111; color: #EEE;
            }
        """)


        controls.addWidget(self.lbl_live)

        self.txt_stats = QtWidgets.QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setStyleSheet("""
            QTextEdit {
                font-family: Consolas, 'Fira Mono', 'DejaVu Sans Mono', monospace;
                font-size: 11pt; background: #0f0f0f; color: #e8e8e8;
                border-radius: 8px; padding: 6px;
            }
        """)
        self.txt_stats.setMinimumHeight(200)

        self.txt_stats.setLineWrapMode(QtWidgets.QTextEdit.WidgetWidth)
        # Prefer wrapping anywhere to avoid long unbreakable tokens widening the panel
        self.txt_stats.setWordWrapMode(QtGui.QTextOption.WrapAnywhere)
        self.txt_stats.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # Don’t let it demand extra horizontal space
        self.txt_stats.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                     QtWidgets.QSizePolicy.Expanding)

        controls.addWidget(self.txt_stats)

        # --- Dataset compute & print -------------------------------------------------
        self.btn_compute_ds = QtWidgets.QPushButton("Compute Dataset & Print Distances")
        self.btn_compute_ds.setToolTip(
            "Build a small dataset with the current Studio settings, then run the dataset's print_* distance functions.")
        self.btn_compute_ds.clicked.connect(self._compute_dataset_and_print_distances)
        controls.addWidget(self.btn_compute_ds)

        # --- Dataset size ------------------------------------------------------------
        rowN = QtWidgets.QWidget()
        rowN_l = QtWidgets.QHBoxLayout(rowN);
        rowN_l.setContentsMargins(0, 0, 0, 0)
        rowN_l.addWidget(QtWidgets.QLabel("N per class"))
        self.s_n_per_class = QtWidgets.QSpinBox()
        self.s_n_per_class.setRange(4, 2000)  # adjust upper bound if you like
        self.s_n_per_class.setSingleStep(4)
        self.s_n_per_class.setValue(48)  # your previous default
        rowN_l.addWidget(self.s_n_per_class)
        controls.addWidget(rowN)

        # Base 1
        add_section("Base 1")
        self.s_k1   = ControlRow("k1", 1, 40, self.bases[0].k, is_int=True); self.s_k1.changed.connect(self.update_from_controls)
        self.s_a1   = ControlRow("a1", 0.0, 20.0, self.bases[0].a, 0.1);     self.s_a1.changed.connect(self.update_from_controls)
        self.s_phi1 = ControlRow("φ1°", -180, 180, math.degrees(self.bases[0].phi), 1.0); self.s_phi1.changed.connect(self.update_from_controls)
        self.phaseMode1 = QtWidgets.QComboBox(); self.phaseMode1.addItems(["relative","signed_absolute"]); self.phaseMode1.setCurrentText(self.bases[0].pmode); self.phaseMode1.currentTextChanged.connect(self.update_from_controls)
        self.s_sphi1 = ControlRow("sφ1", 0.0, 1.0, self.bases[0].pstr, 0.01); self.s_sphi1.changed.connect(self.update_from_controls)
        self.s_Kphi1 = ControlRow("Kφ1", 0.01, 50.0, self.bases[0].pdiv, 0.01); self.s_Kphi1.changed.connect(self.update_from_controls)
        self.ampMode1 = QtWidgets.QComboBox(); self.ampMode1.addItems(["relative","signed_absolute"]); self.ampMode1.setCurrentText(self.bases[0].amode); self.ampMode1.currentTextChanged.connect(self.update_from_controls)
        self.s_sA1 = ControlRow("sA1", 0.0, 1.0, self.bases[0].astr, 0.01); self.s_sA1.changed.connect(self.update_from_controls)
        self.s_KA1 = ControlRow("KA1", 0.01, 50.0, self.bases[0].adiv, 0.01); self.s_KA1.changed.connect(self.update_from_controls)
        for w in (self.s_k1, self.s_a1, self.s_phi1, self.s_sphi1, self.s_Kphi1, self.s_sA1, self.s_KA1): controls.addWidget(w)
        for lab, cmb in (("phase mode 1", self.phaseMode1), ("amp mode 1", self.ampMode1)):
            row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel(lab)); row.addWidget(cmb); row.addStretch(1)
            w = QtWidgets.QWidget(); w.setLayout(row); controls.addWidget(w)

        # Base 2
        add_section("Base 2")
        self.s_k2   = ControlRow("k2", 1, 40, self.bases[1].k, is_int=True); self.s_k2.changed.connect(self.update_from_controls)
        self.s_a2   = ControlRow("a2", 0.0, 20.0, self.bases[1].a, 0.1);     self.s_a2.changed.connect(self.update_from_controls)
        self.s_phi2 = ControlRow("φ2°", -180, 180, math.degrees(self.bases[1].phi), 1.0); self.s_phi2.changed.connect(self.update_from_controls)
        self.phaseMode2 = QtWidgets.QComboBox(); self.phaseMode2.addItems(["relative","signed_absolute"]); self.phaseMode2.setCurrentText(self.bases[1].pmode); self.phaseMode2.currentTextChanged.connect(self.update_from_controls)
        self.s_sphi2 = ControlRow("sφ2", 0.0, 1.0, self.bases[1].pstr, 0.01); self.s_sphi2.changed.connect(self.update_from_controls)
        self.s_Kphi2 = ControlRow("Kφ2", 0.01, 50.0, self.bases[1].pdiv, 0.01); self.s_Kphi2.changed.connect(self.update_from_controls)
        self.ampMode2 = QtWidgets.QComboBox(); self.ampMode2.addItems(["relative","signed_absolute"]); self.ampMode2.setCurrentText(self.bases[1].amode); self.ampMode2.currentTextChanged.connect(self.update_from_controls)
        self.s_sA2 = ControlRow("sA2", 0.0, 1.0, self.bases[1].astr, 0.01); self.s_sA2.changed.connect(self.update_from_controls)
        self.s_KA2 = ControlRow("KA2", 0.01, 50.0, self.bases[1].adiv, 0.01); self.s_KA2.changed.connect(self.update_from_controls)
        for w in (self.s_k2, self.s_a2, self.s_phi2, self.s_sphi2, self.s_Kphi2, self.s_sA2, self.s_KA2): controls.addWidget(w)
        for lab, cmb in (("phase mode 2", self.phaseMode2), ("amp mode 2", self.ampMode2)):
            row = QtWidgets.QHBoxLayout(); row.addWidget(QtWidgets.QLabel(lab)); row.addWidget(cmb); row.addStretch(1)
            w = QtWidgets.QWidget(); w.setLayout(row); controls.addWidget(w)

        # Tiny harmonics (kept; sliders remain)
        add_section("Tiny harmonics")
        self.tiny_rows = []
        for i, T in enumerate(self.tinys):
            lab = QtWidgets.QLabel(f"Tiny {i+1}"); f = lab.font(); f.setBold(True); lab.setFont(f); controls.addWidget(lab)
            r_k = ControlRow(f"k{i+3}", 1, 40, T.k, is_int=True); r_k.changed.connect(self.update_from_controls)
            r_a = ControlRow(f"a{i+3}", 0.0, 20.0, T.a, 0.1);      r_a.changed.connect(self.update_from_controls)
            r_phi = ControlRow(f"φ{i+3}°", -180, 180, math.degrees(T.phi), 1.0); r_phi.changed.connect(self.update_from_controls)
            r_w = ControlRow(f"w{i+3}", 0.0, 1.0, T.weight, 0.01); r_w.changed.connect(self.update_from_controls)
            controls.addWidget(r_k); controls.addWidget(r_a); controls.addWidget(r_phi); controls.addWidget(r_w)
            self.tiny_rows.append((r_k, r_a, r_phi, r_w))

        controls.addStretch(1)
        root.addWidget(ctrlScroll, 0)

        # Right: main plots + exemplars
        right = QtWidgets.QWidget(); right_lay = QtWidgets.QVBoxLayout(right)
        top = QtWidgets.QWidget(); top_lay = QtWidgets.QHBoxLayout(top); right_lay.addWidget(top, 1)
        try:
            self.chk_hide_top.toggled.connect(lambda b: top.setVisible(not b))
        except Exception:
            pass
        self.plot0 = pg.PlotWidget(title="Class 0"); self.plot0.setAspectLocked(True, 1); self.plot0.showGrid(x=True, y=True, alpha=0.2)
        self.curve0 = self.plot0.plot([], [], pen=pg.mkPen('white', width=3.5)); self.dot0 = pg.ScatterPlotItem([0],[0], pen=None, brush=pg.mkBrush('white'), size=8); self.plot0.addItem(self.dot0); self.fill0=None
        self.plot1 = pg.PlotWidget(title="Class 1"); self.plot1.setAspectLocked(True, 1); self.plot1.showGrid(x=True, y=True, alpha=0.2)
        self.curve1 = self.plot1.plot([], [], pen=pg.mkPen('white', width=3.5)); self.dot1 = pg.ScatterPlotItem([0],[0], pen=None, brush=pg.mkBrush('white'), size=8); self.plot1.addItem(self.dot1); self.fill1=None
        top_lay.addWidget(self.plot0,1); top_lay.addWidget(self.plot1,1)

        grid_box = QtWidgets.QGroupBox("Exemplars: Class 0 (top) and closest Class 1 (bottom)")
        grid_lay = QtWidgets.QVBoxLayout(grid_box)
        self.ex_grid = QtWidgets.QGridLayout(); self.ex_rows=[[],[]]; self.ex_pws=[[],[]]; self.ex_fills=[[None]*8,[None]*8]
        for row in range(2):
            for col in range(8):
                pw = pg.PlotWidget(); pw.setAspectLocked(True,1); pw.hideAxis('left'); pw.hideAxis('bottom'); pw.setMenuEnabled(False)
                curve = pw.plot([], [], pen=pg.mkPen('white', width=3.0))
                self.ex_rows[row].append(curve); self.ex_pws[row].append(pw); self.ex_grid.addWidget(pw, row, col)
        grid_lay.addLayout(self.ex_grid)
        h = QtWidgets.QHBoxLayout()

        # NEW: bottom-row toggle
        self.chk_bottom_nearest = QtWidgets.QCheckBox("Bottom row: show nearest matches")
        self.chk_bottom_nearest.setChecked(True)
        self.chk_bottom_nearest.toggled.connect(lambda _: self.update_exemplar_views())

        self.btn_resample = QtWidgets.QPushButton("Resample 8 exemplars")
        self.btn_resample.clicked.connect(self.resample_exemplars)

        h.addWidget(self.chk_bottom_nearest)
        h.addStretch(1)
        h.addWidget(self.btn_resample)
        grid_lay.addLayout(h)

        right_lay.addWidget(grid_box, 1)

        btns = QtWidgets.QHBoxLayout(); self.btn_reset = QtWidgets.QPushButton("Reset"); self.btn_reset.clicked.connect(self.reset)
        self.btn_save = QtWidgets.QPushButton("Export PNGs"); self.btn_save.clicked.connect(self.export_pngs)
        btns.addStretch(1); btns.addWidget(self.btn_reset); btns.addWidget(self.btn_save); right_lay.addLayout(btns)

        root.addWidget(right, 1)

        # Initial sampling & render
        self._exemplar_specs = None
        self._closest_specs = None
        self._exemplar_specs_bottom_random = None  # NEW: bottom row when not showing nearest

        # Debounce timer for expensive exemplar pairing
        self._debounce = QtCore.QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(self._update_exemplar_views_impl)
        self.resample_exemplars()
        self.update_plots()

    def _compute_dataset_and_print_distances(self):
        import io, sys, traceback, re
        self.btn_compute_ds.setEnabled(False)
        self.btn_compute_ds.setText("Computing…")


        try:
            # 1) Import dataset module & push current Studio settings
            from shape_deform_dataset_v35_fixed_json_only_sampler import (
                ShapeDeformDataset,
                print_latent_distance_stats,
                print_image_distance_stats,
            )

            n = int(self.s_n_per_class.value())

            # 2) Build a small dataset (fast) that still exercises the stats
            ds = ShapeDeformDataset(
                nA=n, nB=n,  # adjust if you want denser stats; 48/48 is quick
                image_size=128, intensity=1.0, bg=0.0, norm="none",
                seed=123, batch=256
            )

            # A) Per-class mean intensity and L2 norms (should be ~equal across classes)
            import numpy as np
            X = ds.images.reshape(len(ds.images), -1).astype(np.float32)
            y = ds.labels.astype(int)
            l2 = np.linalg.norm(X, axis=1)
            mu = X.mean(axis=1)
            print("[diag] L2 mean C0/C1:", l2[y == 0].mean(), l2[y == 1].mean())
            print("[diag] μ intensity C0/C1:", mu[y == 0].mean(), mu[y == 1].mean())

            # B) Turn off edges to see if the gap collapses (set both flags same)

            # 3) Capture the printers' stdout
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            try:
                print_latent_distance_stats(ds)
                print_image_distance_stats(ds)
            finally:
                sys.stdout = old_stdout

            text = buf.getvalue()
            self.txt_stats.setText(text)

            # 4) Extract headline numbers (latent block and pixel space block) and paint the big label

            # Find all (within, between) pairs in order of appearance
            pairs = re.findall(
                r"Overall within:\s*([0-9.]+).*?Between.*?:\s*([0-9.]+)",
                text, flags=re.S
            )

            # Choose which block you want:
            #   index 0 = first (usually LATENT), index 1 = second (usually PIXEL/IMAGE)
            if len(pairs) >= 2:
                within_str, between_str = pairs[1]  # <-- PIXEL space
            elif len(pairs) == 1:
                within_str, between_str = pairs[0]  # fallback if only one block printed
            else:
                raise ValueError("Could not parse 'Overall within'/'Between' lines from output.")

            m_within = float(within_str)
            m_between = float(between_str)

            if m_within and m_between:
                within = float(m_within)
                between = float(m_between)
                ratio = (between / within) if within > 1e-12 else 0.0
                bg, fg = "#721c24", "#f8d7da"  # weak
                self.lbl_live.setStyleSheet(
                    f"QLabel {{ font-size: 15pt; font-weight: 800; padding: 8px 12px; "
                    f"border-radius: 12px; background: {bg}; color: {fg}; }}"
                )
                self.lbl_live.setText(
                    f"W  {within:.3f} | B  {between:.3f} "
                )
            else:
                self.lbl_live.setStyleSheet(
                    "QLabel { font-size: 15pt; font-weight: 800; padding: 8px 12px; "
                    "border-radius: 12px; background: #856404; color: #fff3cd; }"
                )
                self.lbl_live.setText("Printed stats parsed, but no summary lines found")

        except Exception as e:
            self.txt_stats.setText(traceback.format_exc())
            self.lbl_live.setStyleSheet(
                "QLabel { font-size: 15pt; font-weight: 800; padding: 8px 12px; "
                "border-radius: 12px; background: #721c24; color: #f8d7da; }"
            )
            self.lbl_live.setText(f"Dataset compute failed: {e}")

        finally:
            self.btn_compute_ds.setEnabled(True)
            self.btn_compute_ds.setText("Compute Dataset & Print Distances")

    def _update_live_stats(self):
        try:
            # Use the dataset’s helper that wraps your two print_* functions
            from shape_deform_dataset_v35_fixed_json_only_sampler import get_latent_and_image_stats_text_for_studio
            text = get_latent_and_image_stats_text_for_studio(self, n_per_class=24, seed=12345)
        except Exception as e:
            self.lbl_live.setStyleSheet(
                "QLabel { font-size: 15pt; font-weight: 800; padding: 8px 12px; border-radius: 12px; background: #721c24; color: #f8d7da; }")
            self.lbl_live.setText(f"Stats error: {e}")
            self.txt_stats.setText(str(e))
            return

        # Show full raw text in the panel
        self.txt_stats.setText(text)

        # Pull the 2 summary numbers from each block (Overall within / Between)
        import re
        def _grab(pattern):
            m = re.search(pattern, text)
            return float(m.group(1)) if m else 0.0

        lat_within = _grab(r"Overall within:\s*([0-9.]+)")
        lat_between = _grab(r"Between.*:\s*([0-9.]+)")
        # for the second (pixel) block, take the next pair of numbers if you want both;
        # or keep the first pair as your headline metric. Here we just show latent as the header:
        ratio = (lat_between / lat_within) if lat_within > 1e-12 else 0.0

        # color badge by ratio (tweak thresholds)
        if ratio >= 1.50:
            bg, fg = "#155724", "#d4edda"
        elif ratio >= 1.20:
            bg, fg = "#856404", "#fff3cd"
        else:
            bg, fg = "#721c24", "#f8d7da"

        self.lbl_live.setStyleSheet(f"""
            QLabel {{
                font-size: 15pt; font-weight: 400; padding: 8px 12px;
                border-radius: 12px; background: {bg}; color: {fg};
            }}
        """)
        self.lbl_live.setText(
            f"W  {lat_within:.3f} | B {lat_between:.3f}"
        )

    def compute_dataset_stats_and_show(self):
        try:
            from shape_deform_dataset_v35_fixed_json_only_sampler import get_latent_and_image_stats_text_for_studio
        except Exception as e:
            self.txt_stats.setText(f"Import error: {e}\nCheck PYTHONPATH / sys.path to reach the dataset module.")
            return

        n = int(self.s_n_stats.value())
        try:
            text = get_latent_and_image_stats_text_for_studio(self, n_per_class=n, seed=12345)
        except Exception as e:
            text = f"Error while computing stats via dataset code:\n{e}"

        self.txt_stats.setText(text)

    def update_from_controls(self):
        if getattr(self, "_ui_guard", False): return
        self._ui_guard = True
        try:
            self.s_arc_phi.slider.setMaximum(max(1,int(self.s_m_phase.value())) - 1)
            self.s_arc_amp.slider.setMaximum(max(1,int(self.s_m_amp.value())) - 1)

            self.m_phase = self.s_m_phase.value(); self.m_amp = self.s_m_amp.value(); self.gap = self.s_gap.value(); self.phase_deg = self.s_phase.value()
            self.which_arc_phi = max(0, min(int(self.s_arc_phi.value()), max(1,int(self.m_phase))-1)); self.pos_phi = self.s_pos_phi.value()
            self.which_arc_amp = max(0, min(int(self.s_arc_amp.value()), max(1,int(self.m_amp))-1)); self.pos_amp = self.s_pos_amp.value()
            self.R = self.s_R.value(); self.profile = self.profile_combo.currentText(); self.sharp = self.s_sharp.value()
            # freq sampling
            self.k_max = self.s_kmax.value(); self.m_freq = self.s_m_freq.value(); self.gap_freq = self.s_gap_freq.value()

            self.bases[0].k = self.s_k1.value(); self.bases[0].a = self.s_a1.value(); self.bases[0].phi = math.radians(self.s_phi1.value())
            self.bases[0].pmode = self.phaseMode1.currentText(); self.bases[0].pstr = self.s_sphi1.value(); self.bases[0].pdiv = self.s_Kphi1.value()
            self.bases[0].amode = self.ampMode1.currentText();   self.bases[0].astr = self.s_sA1.value();  self.bases[0].adiv = self.s_KA1.value()

            self.bases[1].k = self.s_k2.value(); self.bases[1].a = self.s_a2.value(); self.bases[1].phi = math.radians(self.s_phi2.value())
            self.bases[1].pmode = self.phaseMode2.currentText(); self.bases[1].pstr = self.s_sphi2.value(); self.bases[1].pdiv = self.s_Kphi2.value()
            self.bases[1].amode = self.ampMode2.currentText();   self.bases[1].astr = self.s_sA2.value();  self.bases[1].adiv = self.s_KA2.value()

            for i,(r_k,r_a,r_phi,r_w) in enumerate(self.tiny_rows):
                self.tinys[i].k = r_k.value(); self.tinys[i].a = r_a.value(); self.tinys[i].phi = math.degrees(r_phi.value()); self.tinys[i].phi = math.radians(self.tinys[i].phi); self.tinys[i].weight = r_w.value()

            self.kmin1 = self.s_kmin1.value()
            self.kmin2 = self.s_kmin2.value()
            self.k_max = max(self.s_kmax.value(), self.kmin2)  # keep k_max ≥ kmin2
            self.m_freq = self.s_m_freq.value()
            self.gap_freq = self.s_gap_freq.value()

            # Keep amp_min ≤ amp_max and both reasonable vs R
            self.amp_min = self.s_amp_min.value()
            self.amp_max = self.s_amp_max.value()
            if self.amp_min > self.amp_max:
                self.amp_min, self.amp_max = self.amp_max, self.amp_min  # simple swap

            # (optional UX) ensure the sliders reflect the swap
            self.s_amp_min.set_value(self.amp_min)
            self.s_amp_max.set_value(self.amp_max)

            # Re-render plots as you already do...
            # self.update_plots()

            # Debounced metrics refresh (200 ms after last change)
            self._stats_timer.start(250)

            self.update_plots()
        finally:
            self._ui_guard = False

    def on_source_change(self, *_):
        self.resample_exemplars()
        self.update_plots()

    def collect_harmonics(self, cls: int):
        phi_ring = ring_phase_for_class(cls, self.m_phase, self.gap, self.phase_deg, self.which_arc_phi, self.pos_phi,self.PARITY_FLIP)
        a_ring = amp_from_arc(cls, self.m_amp, self.gap, self.amp_min, self.amp_max, self.which_arc_amp, self.pos_amp,self.PARITY_FLIP)

        H = []
        for B in self.bases:
            phi = apply_phase_offset(phi_ring, B.phi, B.pmode, B.pstr, B.pdiv)
            a   = apply_amp_offset(a_ring,  B.a,  B.amode, B.astr, B.adiv)
            H.append(("base", B.k, a, phi))
        for T in self.tinys:
            H.append(("tiny", T.k, T.a, T.phi, T.weight))
        return H, phi_ring

    def render_shape(self, cls: int):
        H, phi_ring = self.collect_harmonics(cls)
        if self.profile == "absolute":
            r = np.full_like(self.thetas, float(self.R), dtype=np.float64)
        else:
            r = np.ones_like(self.thetas, dtype=np.float64)

        for t in H:
            if t[0] == "base":
                _, k, a, phi = t
                a_base = a * (1.0 + (self.sharp if cls == 1 else 0.0))
                r += a_base * np.cos(int(k) * (self.thetas - phi))
            else:
                _, k, a, phi, w = t
                sign = (1.0 if cls == 1 else -1.0)
                a_eff = a * (1.0 + sign * self.sharp * w)
                r += a_eff * np.cos(int(k) * (self.thetas - phi))

        x = r * np.cos(self.thetas); y = r * np.sin(self.thetas)
        return x, y, phi_ring, r

    def _make_fill_item(self, plot_widget, x, y):
        path = QtGui.QPainterPath()
        if len(x) == 0:
            return None
        path.moveTo(x[0], y[0])
        for i in range(1, len(x)):
            path.lineTo(x[i], y[i])
        path.closeSubpath()
        item = QtWidgets.QGraphicsPathItem(path)
        item.setBrush(QtGui.QBrush(QtCore.Qt.white))
        item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        plot_widget.addItem(item)
        item.setZValue(-1)
        return item

    def _update_fill_item(self, existing_item, plot_widget, x, y):
        if existing_item is None:
            return self._make_fill_item(plot_widget, x, y)
        path = QtGui.QPainterPath()
        if len(x) == 0:
            # clear path
            existing_item.setPath(path)
            return existing_item
        path.moveTo(x[0], y[0])
        for i in range(1, len(x)):
            path.lineTo(x[i], y[i])
        path.closeSubpath()
        existing_item.setPath(path)
        return existing_item

    # ----- exemplar helpers -----
    def _render_with_spec(self, cls: int, spec: dict):
        old_phi_idx, old_phi_pos = self.which_arc_phi, self.pos_phi
        old_amp_idx, old_amp_pos = self.which_arc_amp, self.pos_amp
        old_k1, old_k2 = self.bases[0].k, self.bases[1].k
        try:

            if "phi" in spec and spec["phi"] is not None:
                idx, t = spec["phi"]; self.which_arc_phi, self.pos_phi = int(idx), float(t)
            if "amp" in spec and spec["amp"] is not None:
                idx, t = spec["amp"]; self.which_arc_amp, self.pos_amp = int(idx), float(t)
            if "freq" in spec and spec["freq"] is not None:
                k1, k2 = spec["freq"]; self.bases[0].k, self.bases[1].k = int(k1), int(k2)
            return self.render_shape(cls)
        finally:
            self.which_arc_phi, self.pos_phi = old_phi_idx, old_phi_pos
            self.which_arc_amp, self.pos_amp = old_amp_idx, old_amp_pos
            self.bases[0].k, self.bases[1].k = old_k1, old_k2

    #def _distance_xy(self, xa, ya, xb, yb):
    #    return float(((xa - xb)**2 + (ya - yb)**2).mean())

    # Replace your _distance_xy with a downsampled version
    def _distance_xy(self, xa, ya, xb, yb):
        s = 2  # try 4 or 5; smaller = faster
        xa, ya = xa[::s], ya[::s]
        xb, yb = xb[::s], yb[::s]
        return float(((xa - xb) ** 2 + (ya - yb) ** 2).mean())

    def _candidate_grid(self, spec0: dict):
        cand = {}
        if spec0.get("phi") is not None:
            m = max(1, int(self.m_phase)); imax = min(m, 8)
            pos = (0.35, 0.65)   # was (0.2, 0.5, 0.8)
            cand["phi"] = [(i,t) for i in range(imax) for t in pos]
        if spec0.get("amp") is not None:
            m = max(1, int(self.m_amp));   imax = min(m, 8)
            pos = (0.35, 0.65)  # was (0.2, 0.5, 0.8)
            cand["amp"] = [(i,t) for i in range(imax) for t in pos]
        if spec0.get("freq") is not None:
            k1, k2 = spec0["freq"]
            ks=[]
            for dk1 in (-1,0,1):  # was (-2,-1,0,1,2)
                for dk2 in (-1,0,1):     # was (-2,-1,0,1,2)
                    ks.append((max(1,k1+dk1), max(1,k2+dk2)))
            cand["freq"] = ks
        return cand

    def _nearest_for_class1(self, spec0: dict):
        # 1) coarse grid
        # --- coarse grid around spec0 (fast) ---
        x0, y0, *_ = self._render_with_spec(0, spec0)

        # Coarse candidates for phase/amp
        def _phi_neighbors():
            # Seed from spec if present; else from current UI controls
            if "phi" in spec0 and spec0["phi"] is not None:
                i, t = spec0["phi"]
            else:
                i, t = int(self.which_arc_phi), float(self.pos_phi)
            i = max(0, min(int(self.m_phase) - 1, int(i)))
            ts = [max(0.0, t - 0.15), t, min(1.0, t + 0.15)]
            return [{"phi": (i, tt)} for tt in ts]

        def _amp_neighbors():
            if "amp" in spec0 and spec0["amp"] is not None:
                i, t = spec0["amp"]
            else:
                i, t = int(self.which_arc_amp), float(self.pos_amp)
            i = max(0, min(int(self.m_amp) - 1, int(i)))
            ts = [max(0.0, t - 0.15), t, min(1.0, t + 0.15)]
            return [{"amp": (i, tt)} for tt in ts]

        coarse_phi_specs = _phi_neighbors()
        coarse_amp_specs = _amp_neighbors()

        # Coarse candidates for frequency — derive from spec0 if present, else no freq search
        # --- before (current) ---
        # kpair0 = spec0.get("freq", None)
        # if kpair0 is None:
        #     coarse_freq = [None]
        # else:
        #     k10, k20 = int(kpair0[0]), int(kpair0[1])
        #     coarse_freq = [(max(1, k10 + dk1), max(1, k20 + dk2))
        #                    for dk1 in (-1, 0, 1) for dk2 in (-1, 0, 1)]

        # --- after (always search freq) ---
        # Seed the grid from the spec if present; otherwise from current base ks
        kpair0 = spec0.get("freq", None)
        if kpair0 is None:
            k10, k20 = int(self.bases[0].k), int(self.bases[1].k)
        else:
            k10, k20 = int(kpair0[0]), int(kpair0[1])

        # Respect your ring floor/ceiling when freq is “None (ring)”
        kmin1, kmin2 = self.kmin1, self.kmin2  # keep your existing minima or expose as sliders
        kmax = max(self.kmin1, self.kmin2, int(self.s_kmax.value()))

        def _clamp_k(k, kmin):
            return max(kmin, min(kmax, int(k)))

        coarse_freq = [(_clamp_k(k10 + dk1, kmin1), _clamp_k(k20 + dk2, kmin2))
                       for dk1 in (-1, 0, 1) for dk2 in (-1, 0, 1)]

        best, bestd = None, 1e9
        for phs in coarse_phi_specs:  # dicts like {"phi": (i,t)}
            for ams in coarse_amp_specs:  # dicts like {"amp": (i,t)}
                for fr in coarse_freq:  # tuples like (k1,k2)
                    spec1 = {}
                    spec1.update(phs)
                    spec1.update(ams)
                    spec1["freq"] = fr
                    x1, y1, *_ = self._render_with_spec(1, spec1)
                    d = self._distance_xy(x0, y0, x1, y1)
                    if d < bestd:
                        bestd, best = d, spec1

        # 2) local refine around best
        def neighborhood(spec1):
            out = []
            # small local tweaks around the winning arc/pos
            if "phi" in spec1:
                i, t = spec1["phi"];
                out += [{"phi": (i, t2)} for t2 in (max(0.0, t - 0.15), t, min(1.0, t + 0.15))]
            if "amp" in spec1:
                i, t = spec1["amp"];
                out += [{"amp": (i, t2)} for t2 in (max(0.0, t - 0.15), t, min(1.0, t + 0.15))]
            if "freq" in spec1:
                k1, k2 = spec1["freq"]
                for dk1 in (-1, 0, 1):
                    for dk2 in (-1, 0, 1):
                        out += [{"freq": (max(1, k1 + dk1), max(1, k2 + dk2))}]
            return out

        for tweak in neighborhood(best):
            spec1 = dict(best);
            spec1.update(tweak)
            x1, y1, *_ = self._render_with_spec(1, spec1)
            d = self._distance_xy(x0, y0, x1, y1)
            if d < bestd:
                bestd, best = d, spec1
        return best, bestd

    def nearest_index_in_pool(self, cls_src: int, spec0: dict,
                              cls_tgt: int, pool_specs: list) -> int:
        """
        Return argmin_j distance_xy( render(cls_src, spec0), render(cls_tgt, pool_specs[j]) ).
        Pool index is relative to pool_specs (0..len(pool_specs)-1).
        """
        import math
        # render source boundary once
        x0, y0, *_ = self._render_with_spec(cls_src, spec0)

        best_j = -1
        best_d = math.inf
        for j, spec1 in enumerate(pool_specs):
            x1, y1, *_ = self._render_with_spec(cls_tgt, spec1)
            d = self._distance_xy(x0, y0, x1, y1)
            if d < best_d:
                best_d = d
                best_j = j
        return best_j if best_j >= 0 else 0

    # ---- add once in Studio init section if not present ----
    def set_seed(self, seed: int):
        import numpy as _np
        self._rng = _np.random.default_rng(int(seed))

    # ---- replacement for resample_exemplars ----
    def resample_exemplars(self):
        """
        Build a small exemplar pool for the dataset path.
        IMPORTANT: sample phase/amp over the FULL range of per-class arcs,
        not from the UI's 'which arc' sliders.
        """
        import numpy as np
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = np.random.default_rng()  # unseeded fallback

        specs = []
        phase_none = (self.cmb_phase_src.currentIndex() == 1)  # "None (ring)"
        amp_none = (self.cmb_amp_src.currentIndex() == 1)
        freq_none = (self.cmb_freq_src.currentIndex() == 1)

        mphi = max(1, int(self.m_phase))
        mamp = max(1, int(self.m_amp))

        for _ in range(8):
            spec = {}

            # ---- PHASE (local arc index, t in [0,1)) ----
            if phase_none:
                which_arc_phi = int(rng.integers(0, mphi, endpoint=False))  # 0..mphi-1
                t_phi = float(rng.random())
                spec["phi"] = (which_arc_phi, t_phi)

            # ---- AMP (local band index, t in [0,1)) ----
            if amp_none:
                which_arc_amp = int(rng.integers(0, mamp, endpoint=False))  # 0..mamp-1
                t_amp = float(rng.random())
                spec["amp"] = (which_arc_amp, t_amp)

            # ---- FREQ (draw within arc sectors; keep your logic, but use rng) ----
            if freq_none:
                kmin1, kmin2 = self.kmin1, self.kmin2
                kmax = max(self.kmin1, self.kmin2, int(self.s_kmax.value()))
                m = max(1, int(self.s_m_freq.value()))
                total = 2 * m
                # choose an arc index for this exemplar (shared by k1/k2 for coherence)
                arc_idx = int(rng.integers(0, m, endpoint=False))  # 0..m-1
                sector_width = (kmax - kmin1 + 1) / float(total)
                sector_width2 = (kmax - kmin2 + 1) / float(total)
                usable = (1.0 - float(self.s_gap_freq.value()))
                # even sector for class-0 parity; class-1 handled later by parity add
                sec0 = 2 * arc_idx
                base0 = kmin1 + sec0 * sector_width + 0.5 * float(self.s_gap_freq.value()) * sector_width
                base02 = kmin2 + sec0 * sector_width2 + 0.5 * float(self.s_gap_freq.value()) * sector_width2

                lo = int(max(kmin1, np.ceil(base0)))
                hi = int(min(kmax, np.floor(base0 + usable * sector_width)))
                if lo > hi: lo, hi = kmin1, kmax
                k1 = int(rng.integers(lo, hi + 1))  # inclusive hi

                lo2 = int(max(kmin2, np.ceil(base02)))
                hi2 = int(min(kmax, np.floor(base02 + usable * sector_width2)))
                if lo2 > hi2: lo2, hi2 = kmin2, kmax
                k2 = int(rng.integers(lo2, hi2 + 1))

                spec["freq"] = (k1, k2)

            # (optional) keep local indices for debugging
            # they are already in spec["phi"] and spec["amp"] as (local_idx, t)
            specs.append(spec)

            self._exemplar_specs = specs

            # NEW: build an independent bottom-row draw using the same sampling mode as top
            rng2 = getattr(self, "_rng", None) or __import__("numpy").random.default_rng()
            self._exemplar_specs_bottom_random = [self._resample_like_top(s, rng2) for s in specs]

            self._closest_specs = [None] * 8
            self.update_exemplar_views()

    def _resample_like_top(self, spec0: dict, rng):
        """
        Make a new spec that follows the same 'None (ring)' vs 'Fixed' logic as the top row,
        but as an independent draw. We reuse spec0's chosen arc indices, and just re-draw the
        local t-values (and freq if 'None (ring)').
        """
        import copy, math, numpy as np
        s = copy.deepcopy(spec0)

        # Re-draw local t in [0,1) for phase/amp if present
        if isinstance(s.get("phi"), (tuple, list)) and len(s["phi"]) == 2:
            i_phi, _t = s["phi"]
            s["phi"] = (i_phi, float(rng.random()))
        if isinstance(s.get("amp"), (tuple, list)) and len(s["amp"]) == 2:
            i_amp, _t = s["amp"]
            s["amp"] = (i_amp, float(rng.random()))

        # If frequency was sampled from ring (“None”), draw a new (k1,k2) in same admissible range.
        # If it was Fixed, keep it identical.
        # We infer “None” vs “Fixed” from current combo boxes.
        freq_none = (self.cmb_freq_src.currentIndex() == 1)  # "None (ring)"
        if freq_none and isinstance(s.get("freq"), (tuple, list)) and len(s["freq"]) == 2:
            k1_min = int(getattr(self, "kmin1", 1))
            k2_min = int(getattr(self, "kmin2", 1))
            # Upper bound comes from the same slider you use elsewhere
            kmax_slider = int(self.s_kmax.value()) if hasattr(self, "s_kmax") else max(k1_min, k2_min, 8)
            k1_max = int(max(k1_min, kmax_slider))
            k2_max = int(max(k2_min, kmax_slider))

            s["freq"] = (
                int(rng.integers(k1_min, k1_max + 1)),
                int(rng.integers(k2_min, k2_max + 1)),
            )

        return s


    def update_exemplar_views(self):
        # Debounce heavy recompute
        self._debounce.start()

    def _update_exemplar_views_impl(self):
        if not self._exemplar_specs: return
        for col, spec0 in enumerate(self._exemplar_specs):
            x0, y0, *_ = self._render_with_spec(0, spec0)
            self.ex_rows[0][col].setData(x0,y0)
            # Update fill for top exemplar
            self.ex_fills[0][col] = self._update_fill_item(self.ex_fills[0][col], self.ex_pws[0][col], x0, y0)
            best,_ = self._nearest_for_class1(spec0)
            self._closest_specs[col] = best
            x1, y1, *_ = self._render_with_spec(1, best or {})
            self.ex_rows[1][col].setData(x1,y1)
            # BOTTOM ROW: either nearest matches (checkbox ON) or independent exemplars sampled like top (checkbox OFF)
            if getattr(self, "chk_bottom_nearest", None) is not None and self.chk_bottom_nearest.isChecked():
                # Show nearest matches (existing behavior)
                best, _ = self._nearest_for_class1(spec0)
                self._closest_specs[col] = best
                target_spec = best if best else spec0  # fall back to a valid spec
            else:
                # Show independent exemplars sampled like the top
                if not self._exemplar_specs_bottom_random or len(self._exemplar_specs_bottom_random) < len(
                        self._exemplar_specs):
                    # Safety: if toggle was flipped before a resample, build the list now
                    rng2 = getattr(self, "_rng", None) or __import__("numpy").random.default_rng()
                    self._exemplar_specs_bottom_random = [self._resample_like_top(s, rng2) for s in
                                                          self._exemplar_specs]
                target_spec = self._exemplar_specs_bottom_random[col]

            x1, y1, *_ = self._render_with_spec(1, target_spec)
            self.ex_rows[1][col].setData(x1, y1)
            self.ex_fills[1][col] = self._update_fill_item(self.ex_fills[1][col], self.ex_pws[1][col], x1, y1)

    def update_plots(self):
        x0, y0, phi0, _ = self.render_shape(0); x1, y1, phi1, _ = self.render_shape(1)
        self.curve0.setData(x0,y0); self.curve1.setData(x1,y1)
        # Update white fills
        self.fill0 = self._update_fill_item(self.fill0, self.plot0, x0, y0)
        self.fill1 = self._update_fill_item(self.fill1, self.plot1, x1, y1)
        self.dot0.setData([math.cos(phi0)],[math.sin(phi0)]); self.dot1.setData([math.cos(phi1)],[math.sin(phi1)])
        base_R = self.R if self.profile=="absolute" else 1.0
        max_add = sum(abs(B.a) for B in self.bases) + sum(abs(T.a)*(1.0+abs(self.sharp*T.weight)) for T in self.tinys)
        Rmax = base_R + max_add + 0.2
        for plt in (self.plot0,self.plot1):
            plt.setXRange(-Rmax, Rmax, padding=0); plt.setYRange(-Rmax, Rmax, padding=0)
        self.update_exemplar_views()

    def reset(self):
        self.__init__(); self.show()

    def export_pngs(self):
        exporter = ImageExporter(self.plot0.plotItem); exporter.parameters()['width']=800; exporter.export("shape_class0.png")
        exporter = ImageExporter(self.plot1.plotItem); exporter.parameters()['width']=800; exporter.export("shape_class1.png")
        QtWidgets.QMessageBox.information(self, "Export", "Saved: shape_class0.png, shape_class1.png")

    # --- In ShapeStudio class ---

    def get_ring_config(self):
        """
        Returns a dict of ring parameters used everywhere.
        Extend if you have separate gap/fracs for phase/amp/freq.
        """
        return {
            "m_phase": getattr(self, "m_phase", 2),
            "m_amp": getattr(self, "m_amp", 2),
            "gap_phase": getattr(self, "gap_frac_phase", getattr(self, "gap_frac", 0.0)),
            "gap_amp": getattr(self, "gap_frac_amp", getattr(self, "gap_frac", 0.0)),
            # Optional advanced knobs (only if you actually have them):
            "phase_rad_offset": getattr(self, "phase_rad_offset", 0.0),  # global rotation on phase ring
            "amp_rad_offset": getattr(self, "amp_rad_offset", 0.0),
        }

    def _current_settings_dict(self) -> dict:
        import math
        # Map Studio state → JSON fields that match set_studio_from_main signature
        data = {
            # --- ring topology & global params ---
            "m_phase": int(self.m_phase),
            "m_amp": int(self.m_amp),
            "gap": float(self.gap),
            "phase_deg": float(self.phase_deg),
            "which_arc_phi": int(self.which_arc_phi),
            "pos_phi": float(self.pos_phi),
            "which_arc_amp": int(self.which_arc_amp),
            "pos_amp": float(self.pos_amp),
            "R": float(self.R),
            "profile": str(self.profile),
            "sharp": float(self.sharp),
            "amp_min": float(self.amp_min),
            "amp_max": float(self.amp_max),

            # --- frequency ring controls ---
            "k_min1": int(getattr(self, "kmin1", 2)),
            "k_min2": int(getattr(self, "kmin2", 3)),
            "k_max": int(self.k_max),
            "m_freq": int(self.m_freq),
            "gap_freq": float(self.gap_freq),

            # --- sources (dropdown texts) ---
            "phase_src": self.cmb_phase_src.currentText(),
            "amp_src": self.cmb_amp_src.currentText(),
            "freq_src": self.cmb_freq_src.currentText(),

            # --- base harmonics (convert phases to degrees for readability) ---
            "k1": int(self.bases[0].k),
            "a1": float(self.bases[0].a),
            "phi1_deg": float(math.degrees(self.bases[0].phi)),
            "phase_mode1": str(self.bases[0].pmode),
            "sphi1": float(self.bases[0].pstr),
            "Kphi1": float(self.bases[0].pdiv),
            "amp_mode1": str(self.bases[0].amode),
            "sA1": float(self.bases[0].astr),
            "KA1": float(self.bases[0].adiv),

            "k2": int(self.bases[1].k),
            "a2": float(self.bases[1].a),
            "phi2_deg": float(math.degrees(self.bases[1].phi)),
            "phase_mode2": str(self.bases[1].pmode),
            "sphi2": float(self.bases[1].pstr),
            "Kphi2": float(self.bases[1].pdiv),
            "amp_mode2": str(self.bases[1].amode),
            "sA2": float(self.bases[1].astr),
            "KA2": float(self.bases[1].adiv),

            # --- tiny harmonics: list of (k,a,phi_deg,weight) ---
            "tinys": [
                [int(T.k), float(T.a), float(math.degrees(T.phi)), float(T.weight)]
                for T in self.tinys
            ],
        }
        return data

    def save_settings_to_file(self):
        import json, pathlib
        # make sure there's an app (harmless if one exists)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        default_path = str(pathlib.Path.home() / "studio_settings.json")
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog  # uncomment if native dialog causes issues

        parent = self if isinstance(self, QtWidgets.QWidget) else None
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent, "Save Studio Settings", default_path, "JSON (*.json)", options=options
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._current_settings_dict(), f, indent=2)

    @staticmethod
    def _arc_bounds(total_class_arcs: int, gap_frac: float, rad_offset: float = 0.0):
        """
        Alternating arcs around [0, 2π), even indices = Class 0, odd = Class 1.
        Returns: list of (start,end) for all 2*m arcs, in radians (wrapped to [0, 2π)).
        """
        import math
        total = 2 * int(total_class_arcs)
        base_w = 2 * math.pi / total
        gap = gap_frac * base_w
        arcs = []
        for k in range(total):
            start = rad_offset + k * base_w + gap / 2.0
            end = rad_offset + (k + 1) * base_w - gap / 2.0
            # wrap to [0, 2π)
            two_pi = 2.0 * math.pi
            s = start % two_pi
            e = end % two_pi
            arcs.append((s, e))
        return arcs  # indices: 0..2m-1, parity encodes class

    @staticmethod
    def angle_on_ring(arc_idx: int, t_in_arc: float, total_class_arcs: int, gap_frac: float, rad_offset: float = 0.0):
        """
        Map (arc index, position t in [0,1]) to absolute angle θ on the ring.
        """
        import math
        total = 2 * int(total_class_arcs)
        base_w = 2 * math.pi / total
        gap = gap_frac * base_w
        usable = base_w - gap
        start = rad_offset + arc_idx * base_w + gap / 2.0
        theta = start + float(t_in_arc) * usable
        return (theta + 2.0 * math.pi) % (2.0 * math.pi)

    def get_phase_ring_arcs(self):
        c = self.get_ring_config()
        return self._arc_bounds(c["m_phase"], c["gap_phase"], c.get("phase_rad_offset", 0.0))

    def get_amp_ring_arcs(self):
        c = self.get_ring_config()
        return self._arc_bounds(c["m_amp"], c["gap_amp"], c.get("amp_rad_offset", 0.0))

    def theta_from_spec(self, spec: dict):
        """
        Convenience: extract ring angles (phase and amp) for a given spec.
        Returns (phi_theta, amp_theta) where each may be None if missing in the spec.
        """
        c = self.get_ring_config()
        phi_theta = None
        amp_theta = None
        if isinstance(spec, dict):
            if spec.get("phi") is not None:
                i_phi, t_phi = int(spec["phi"][0]), float(spec["phi"][1])
                phi_theta = self.angle_on_ring(i_phi, t_phi, c["m_phase"], c["gap_phase"],
                                               c.get("phase_rad_offset", 0.0))
            if spec.get("amp") is not None:
                i_amp, t_amp = int(spec["amp"][0]), float(spec["amp"][1])
                amp_theta = self.angle_on_ring(i_amp, t_amp, c["m_amp"], c["gap_amp"], c.get("amp_rad_offset", 0.0))
        return phi_theta, amp_theta


# --- export Studio search for dataset reuse ---
def studio_find_nearest(spec0, studio_instance):
    """Wrapper for dataset use — reuses Studio's contour-based nearest search."""
    return studio_instance._nearest_for_class1(spec0)


def main():
    app = QtWidgets.QApplication(sys.argv); pg.setConfigOptions(antialias=True)
    w = ShapeStudio(); w.show(); sys.exit(app.exec())





if __name__ == "__main__":
    main()
