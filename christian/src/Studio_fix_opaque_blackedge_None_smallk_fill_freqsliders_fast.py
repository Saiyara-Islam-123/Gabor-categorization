#!/usr/bin/env python3
# ShapeStudio (PySide6 + PyQtGraph) — dual-class live preview + exemplar pairing with None/Fixed sources
# Deps: pip install PySide6 pyqtgraph numpy
import math
import sys
from dataclasses import dataclass
import numpy as np

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter

TAU = math.tau if hasattr(math, "tau") else 2*math.pi

def wrap_signed(x: float) -> float:
    return ((x + math.pi) % TAU + TAU) % TAU - math.pi

def ring_phase_for_class(cls: int, m_arcs: int, gap_frac: float, phase_deg: float, which_arc: int, pos_in_arc: float) -> float:
    total = max(1, 2*int(m_arcs))
    base_sector = TAU / total
    usable = (1.0 - gap_frac) * base_sector
    off = math.radians(phase_deg)
    global_sector = (int(which_arc) * 2 + (1 if cls == 1 else 0)) % total
    start = global_sector * base_sector + off + 0.5*gap_frac*base_sector
    pos_in_arc = max(0.0, min(1.0, pos_in_arc))
    phi = start + pos_in_arc * usable
    return (phi % TAU + TAU) % TAU

def amp_from_arc(cls: int, m_arcs: int, gap_frac: float, amp_min: float, amp_max: float, which_arc: int, pos_in_arc: float) -> float:
    total = max(1, 2*int(m_arcs))
    usable = (1.0 - gap_frac)
    band_width = usable / total
    global_band = (int(which_arc) * 2 + (1 if cls==1 else 0)) % total
    lo = amp_min + (global_band * band_width + gap_frac/2.0) * (amp_max - amp_min)
    hi = lo + band_width * (amp_max - amp_min)
    t = max(0.0, min(1.0, pos_in_arc))
    return max(amp_min, min(amp_max, lo + t * (hi - lo)))

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
        self.R = 40.0
        self.profile = "absolute"
        self.sharp = 0.3
        self.amp_min = 0.5; self.amp_max = 1.5
        # Frequency sampling state (None mode)
        self.k_max = 8
        self.m_freq = 2
        self.gap_freq = 0.30
        self.bases = [
            BaseHarm(k=3, a=0.9, phi=math.radians(45), pmode="signed_absolute", pstr=0.25, pdiv=10.0,
                     amode="relative", astr=0.25, adiv=10.0),
            BaseHarm(k=5, a=0.6, phi=0.0, pmode="signed_absolute", pstr=0.25, pdiv=10.0,
                     amode="relative", astr=0.25, adiv=10.0),
        ]
        self.tinys = [
            TinyHarm(k=8, a=0.0, phi=0.0, weight=0.5),
            TinyHarm(k=14, a=0.0, phi=0.0, weight=0.8),
            TinyHarm(k=2, a=0.0, phi=0.0, weight=0.3),
        ]

        self.N = 1600
        self.thetas = np.linspace(0, TAU, self.N, endpoint=False)

        # ----- UI -----
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # Left controls
        ctrlScroll = QtWidgets.QScrollArea(); ctrlScroll.setWidgetResizable(True)
        ctrl = QtWidgets.QWidget(); ctrlScroll.setWidget(ctrl)
        controls = QtWidgets.QVBoxLayout(ctrl)

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
        self.s_kmax = ControlRow("k_max (freq None)", 4, 40, 8, is_int=True); self.s_kmax.changed.connect(self.update_from_controls)
        self.s_m_freq = ControlRow("m_freq (arcs)", 1, 10, 2, is_int=True); self.s_m_freq.changed.connect(self.update_from_controls)
        self.s_gap_freq = ControlRow("gap_freq", 0.0, 0.8, 0.30, 0.01); self.s_gap_freq.changed.connect(self.update_from_controls)
        controls.addWidget(self.s_kmax); controls.addWidget(self.s_m_freq); controls.addWidget(self.s_gap_freq)

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
        h = QtWidgets.QHBoxLayout(); self.btn_resample = QtWidgets.QPushButton("Resample 8 exemplars"); self.btn_resample.clicked.connect(self.resample_exemplars)
        h.addStretch(1); h.addWidget(self.btn_resample); grid_lay.addLayout(h)
        right_lay.addWidget(grid_box, 1)

        btns = QtWidgets.QHBoxLayout(); self.btn_reset = QtWidgets.QPushButton("Reset"); self.btn_reset.clicked.connect(self.reset)
        self.btn_save = QtWidgets.QPushButton("Export PNGs"); self.btn_save.clicked.connect(self.export_pngs)
        btns.addStretch(1); btns.addWidget(self.btn_reset); btns.addWidget(self.btn_save); right_lay.addLayout(btns)

        root.addWidget(right, 1)

        # Initial sampling & render
        self._exemplar_specs = None; self._closest_specs = None
        # Debounce timer for expensive exemplar pairing
        self._debounce = QtCore.QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(30)
        self._debounce.timeout.connect(self._update_exemplar_views_impl)
        self.resample_exemplars()
        self.update_plots()

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

            self.update_plots()
        finally:
            self._ui_guard = False

    def on_source_change(self, *_):
        self.resample_exemplars()
        self.update_plots()

    def collect_harmonics(self, cls: int):
        phi_ring = ring_phase_for_class(cls, self.m_phase, self.gap, self.phase_deg, self.which_arc_phi, self.pos_phi)
        a_ring = amp_from_arc(cls, self.m_amp, self.gap, self.amp_min, self.amp_max, self.which_arc_amp, self.pos_amp)

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
        coarse_phi = [(i, t) for i in range(min(max(1, int(self.m_phase)), 2))
                      for t in (0.33, 0.66)] if ("phi" in spec0) else [None]
        coarse_amp = [(i, t) for i in range(min(max(1, int(self.m_amp)), 2))
                      for t in (0.33, 0.66)] if ("amp" in spec0) else [None]

        # Coarse candidates for frequency — derive from spec0 if present, else no freq search
        kpair0 = spec0.get("freq", None)
        if kpair0 is None:
            coarse_freq = [None]
        else:
            k10, k20 = int(kpair0[0]), int(kpair0[1])
            coarse_freq = [(max(1, k10 + dk1), max(1, k20 + dk2))
                           for dk1 in (-1, 0, 1) for dk2 in (-1, 0, 1)]

        best = None;
        bestd = 1e9
        for ph in (coarse_phi if spec0.get("phi") is not None else [None]):
            for am in (coarse_amp if spec0.get("amp") is not None else [None]):
                for fr in (coarse_freq if spec0.get("freq") is not None else [None]):
                    spec1 = {k: v for k, v in {"phi": ph, "amp": am, "freq": fr}.items() if v is not None}
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

    def resample_exemplars(self):
        specs = []
        phase_none = (self.cmb_phase_src.currentIndex()==1)
        amp_none   = (self.cmb_amp_src.currentIndex()==1)
        freq_none  = (self.cmb_freq_src.currentIndex()==1)
        mphi = max(1,int(self.m_phase)); mamp=max(1,int(self.m_amp))
        import random
        for _ in range(8):
            spec = {}
            if phase_none:
                spec["phi"] = (random.randrange(mphi), random.random())
            if amp_none:
                spec["amp"] = (random.randrange(mamp), random.random())
            if freq_none:
                # Sample integers within [kmin, k_max] using 2*m arcs with gap removal
                kmin1, kmin2 = 2, 3
                kmax = max(kmin2, int(self.s_kmax.value()))
                m = max(1, int(self.s_m_freq.value()))
                total = 2 * m
                # choose an arc index for this exemplar (shared by k1/k2 for coherence)
                arc_idx = random.randrange(m)
                sector_width = (kmax - kmin1 + 1) / float(total)
                usable = (1.0 - float(self.s_gap_freq.value())) * sector_width
                # Class 0 uses even sectors
                sec0 = 2 * arc_idx
                base0 = kmin1 + sec0 * sector_width + 0.5 * float(self.s_gap_freq.value()) * sector_width
                lo = int(max(kmin1, math.ceil(base0)))
                hi = int(min(kmax, math.floor(base0 + usable)))
                if lo > hi:
                    lo, hi = kmin1, kmax
                k1 = random.randint(lo, hi)
                # k2 uses same arc but starts at kmin2
                sector_width2 = (kmax - kmin2 + 1) / float(total)
                base02 = kmin2 + sec0 * sector_width2 + 0.5 * float(self.s_gap_freq.value()) * sector_width2
                lo2 = int(max(kmin2, math.ceil(base02)))
                hi2 = int(min(kmax, math.floor(base02 + usable)))
                if lo2 > hi2:
                    lo2, hi2 = kmin2, kmax
                k2 = random.randint(lo2, hi2)
                spec["freq"] = (k1, k2)
            specs.append(spec)
        self._exemplar_specs = specs
        self._closest_specs  = [None]*8
        self.update_exemplar_views()


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
            # Update fill for bottom exemplar
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
