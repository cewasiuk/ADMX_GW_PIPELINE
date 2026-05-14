# """
# admx_db_datatypes.py
# --------------------
# Core data structures for the ADMX gravitational-wave reanalysis pipeline.
# """

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
MU0 = 4.0 * np.pi * 1e-7      # vacuum permeability  [H/m]
C_LIGHT = 299_792_458.0        # speed of light       [m/s]
K_BOLTZ = 1.380_649e-23        # Boltzmann constant   [J/K]


# ---------------------------------------------------------------------------
# ScanParameters  –  per-scan instrument state
# ---------------------------------------------------------------------------
@dataclass
class ScanParameters:

    # ---- band & cavity geometry ----------------------------------------
    start_freq_hz: float = 0.0
    stop_freq_hz: float = 0.0
    f0_hz: float = 0.0
    quality_factor: float = 1.0
    b_field_tesla: float = 0.0
    coupling: float = 0.0          # β; must be read per-scan, not hardcoded

    # ---- receiver chain ------------------------------------------------
    jpa_snri_db: float = 0.0
    thfet_kelvin: float = 0.0
    attenuation_db: float = 0.0

    # ---- run parameters ------------------------------------------------
    integration_time_s: float = 0.0
    volume_m3: float = 0.136       # ADMX Run 1B: 136 L
    timestamp: str = ""
    filename_tag: str = ""

    # ---- GW analysis ---------------------------------------------------
    eta: float = 0.1               # TM_010 coupling coefficient (Berlin+)
    form_factor: Optional[float] = None

    # ---- derived properties -------------------------------------------
    @property
    def center_freq_hz(self) -> float:
        return 0.5 * (self.start_freq_hz + self.stop_freq_hz)

    @property
    def bandwidth_hz(self) -> float:
        return self.stop_freq_hz - self.start_freq_hz

    @property
    def cavity_linewidth_hz(self) -> float:
        if self.quality_factor <= 0:
            raise ValueError("quality_factor must be positive")
        return self.f0_hz / self.quality_factor

    @property
    def tsys_kelvin(self) -> float:

        snri_linear = 10.0 ** (self.jpa_snri_db / 10.0)
        atten_linear = 10.0 ** (self.attenuation_db / 10.0)
        if snri_linear * atten_linear == 0:
            raise ValueError("SNRI or attenuation is zero; cannot compute T_sys")
        return self.thfet_kelvin / (snri_linear * atten_linear)

    @classmethod
    def from_dataframe_row(cls, row, b_field_tesla: float,
                           volume_m3: float = 0.136,
                           eta: float = 0.1) -> "ScanParameters":
        return cls(
            start_freq_hz=float(row["Start_Frequency"]) * 1e6,
            stop_freq_hz=float(row["Stop_Frequency"]) * 1e6,
            f0_hz=float(row["Cavity_Resonant_Frequency"]) * 1e6,
            quality_factor=float(row["Quality_Factor"]),
            b_field_tesla=float(b_field_tesla),
            coupling=float(row["Reflection"]),     # β from reflection scan
            jpa_snri_db=float(row["JPA_SNR"]),
            thfet_kelvin=float(row["Thfet"]),
            attenuation_db=float(row["Attenuation"]),
            integration_time_s=float(row["Integration_Time"]),
            volume_m3=float(volume_m3),
            filename_tag=str(row["Filename_Tag"]),
            eta=float(eta),
        )

    def __repr__(self) -> str:
        return (
            f"ScanParameters(f0={self.f0_hz/1e6:.3f} MHz, "
            f"Q={self.quality_factor:.0f}, B={self.b_field_tesla:.2f} T, "
            f"β={self.coupling:.3f}, T_sys={self.tsys_kelvin:.2f} K, "
            f"t_int={self.integration_time_s:.1f} s, "
            f"tag='{self.filename_tag}')"
        )


# ---------------------------------------------------------------------------
# ADMXDataSeries  –  base uniformly-sampled series
# ---------------------------------------------------------------------------
class ADMXDataSeries:

    def __init__(
        self,
        yvalues,
        xstart: float,
        xstop: float,
        xunits: str = "Unknown",
        yunits: str = "Unknown",
        metadata: Optional[dict] = None,
    ) -> None:
        self.yvalues: np.ndarray = np.asarray(yvalues)
        self.xstart = float(xstart)
        self.xstop = float(xstop)
        self.xunits = xunits
        self.yunits = yunits
        self.metadata: dict = copy.deepcopy(metadata) if metadata is not None else {}

        if self.xstart >= self.xstop:
            raise ValueError(
                f"xstart ({self.xstart}) must be less than xstop ({self.xstop})"
            )
        if len(self.yvalues) == 0:
            raise ValueError("yvalues must not be empty")

    # ---- copy -----------------------------------------------------------
    def copy(self) -> "ADMXDataSeries":
        return ADMXDataSeries(
            np.copy(self.yvalues), self.xstart, self.xstop,
            xunits=self.xunits, yunits=self.yunits,
            metadata=copy.deepcopy(self.metadata),
        )

    # ---- x-axis helpers -------------------------------------------------
    def get_xspacing(self) -> float:
        """Bin width: (xstop - xstart) / N."""
        return (self.xstop - self.xstart) / len(self.yvalues)

    def get_xvalues(self) -> np.ndarray:
        """Left-edge x coordinate of every bin."""
        return np.linspace(self.xstart, self.xstop, len(self.yvalues),
                           endpoint=False)

    def get_xcentres(self) -> np.ndarray:
        """Centre x coordinate of every bin."""
        return self.get_xvalues() + 0.5 * self.get_xspacing()

    def get_delta_xvalues(self) -> np.ndarray:
        """Offset of bin centres from the series centre."""
        centres = self.get_xcentres()
        mid = 0.5 * (self.xstart + self.xstop)
        return centres - mid

    def get_x_at_index(self, index: int) -> float:
        """Left-edge x coordinate of bin at *index*."""
        return self.xstart + float(index) * self.get_xspacing()

    def get_x_index_below_x(self, xval: float) -> int:
        """Index of the bin whose left edge is <= xval (clamped to [0, N))."""
        idx = int(math.floor((xval - self.xstart) / self.get_xspacing()))
        return max(0, min(idx, len(self.yvalues) - 1))

    def interp_y_at_x(self, xval: float) -> float:
        """
        Linear interpolation of y at xval; clips to edge values outside range.
        """
        if xval <= self.xstart:
            return float(self.yvalues[0])
        if xval >= self.xstop:
            return float(self.yvalues[-1])
        idx = self.get_x_index_below_x(xval)
        # guard against reaching the last bin
        if idx >= len(self.yvalues) - 1:
            return float(self.yvalues[-1])
        x1 = self.get_x_at_index(idx)
        x2 = self.get_x_at_index(idx + 1)
        y1 = float(self.yvalues[idx])
        y2 = float(self.yvalues[idx + 1])
        return y1 + (y2 - y1) * (xval - x1) / (x2 - x1)

    # ---- slicing --------------------------------------------------------
    def subseries(self, starti: int, endi: int) -> "ADMXDataSeries":
        """Return a new series covering bins [starti, endi)."""
        return ADMXDataSeries(
            self.yvalues[starti:endi],
            self.get_x_at_index(starti),
            self.get_x_at_index(endi),
            xunits=self.xunits,
            yunits=self.yunits,
            metadata=copy.deepcopy(self.metadata),
        )

    # ---- arithmetic operators ------------------------------------------
    def _check_compatible(self, other: "ADMXDataSeries") -> None:
        if len(self) != len(other):
            raise ValueError(
                f"Series length mismatch: {len(self)} vs {len(other)}"
            )
        if not (math.isclose(self.xstart, other.xstart, rel_tol=1e-9) and
                math.isclose(self.xstop, other.xstop, rel_tol=1e-9)):
            raise ValueError(
                f"Series x-range mismatch: "
                f"[{self.xstart}, {self.xstop}] vs [{other.xstart}, {other.xstop}]"
            )

    def __add__(self, other):
        if isinstance(other, ADMXDataSeries):
            self._check_compatible(other)
            return self._new_like(self.yvalues + other.yvalues)
        if isinstance(other, (np.ndarray, float, int)):
            return self._new_like(self.yvalues + np.asarray(other))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, ADMXDataSeries):
            self._check_compatible(other)
            return self._new_like(self.yvalues - other.yvalues)
        if isinstance(other, (np.ndarray, float, int)):
            return self._new_like(self.yvalues - np.asarray(other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (np.ndarray, float, int)):
            arr = np.asarray(other)
            if arr.ndim > 0 and len(arr) != len(self):
                raise ValueError(
                    f"Array length {len(arr)} does not match series length {len(self)}"
                )
            return self._new_like(self.yvalues * arr)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (np.ndarray, float, int)):
            arr = np.asarray(other, dtype=float)
            if arr.ndim > 0 and len(arr) != len(self):
                raise ValueError(
                    f"Array length {len(arr)} does not match series length {len(self)}"
                )
            if np.any(arr == 0):
                raise ZeroDivisionError("Division by zero in ADMXDataSeries.__truediv__")
            return self._new_like(self.yvalues / arr)
        return NotImplemented

    def _new_like(self, new_yvalues: np.ndarray) -> "ADMXDataSeries":
        """
        Return a new instance of the same concrete subclass with updated yvalues.
        Subclasses override this to preserve their extra fields.
        """
        obj = ADMXDataSeries(
            new_yvalues, self.xstart, self.xstop,
            xunits=self.xunits, yunits=self.yunits,
            metadata=copy.deepcopy(self.metadata),
        )
        return obj

    # ---- dunder ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.yvalues)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N={len(self)}, "
            f"x=[{self.xstart:.6g}, {self.xstop:.6g}] {self.xunits}, "
            f"y={self.yunits})"
        )

    def __str__(self) -> str:
        lines = [f'# "{self.xunits}" "{self.yunits}"']
        for x, y in zip(self.get_xvalues(), self.yvalues):
            lines.append(f"{x} {y}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PowerSpectrum  –  real power vs frequency
# ---------------------------------------------------------------------------
class PowerSpectrum(ADMXDataSeries):

    def __init__(self, power, start_freq_hz: float, stop_freq_hz: float,
                 metadata: Optional[dict] = None) -> None:
        super().__init__(
            np.asarray(power, dtype=np.float64),
            start_freq_hz, stop_freq_hz,
            xunits="Frequency (Hz)",
            yunits="Power (W)",
            metadata=metadata,
        )

    def copy(self) -> "PowerSpectrum":
        return PowerSpectrum(
            np.copy(self.yvalues), self.xstart, self.xstop,
            metadata=copy.deepcopy(self.metadata),
        )

    def _new_like(self, new_yvalues: np.ndarray) -> "PowerSpectrum":
        return PowerSpectrum(
            new_yvalues, self.xstart, self.xstop,
            metadata=copy.deepcopy(self.metadata),
        )


# ---------------------------------------------------------------------------
# PowerMeasurement  –  power spectrum with per-bin uncertainties
# ---------------------------------------------------------------------------
class PowerMeasurement(PowerSpectrum):


    def __init__(self, power, power_unc, start_freq_hz: float,
                 stop_freq_hz: float, metadata: Optional[dict] = None) -> None:
        super().__init__(power, start_freq_hz, stop_freq_hz, metadata=metadata)
        self.yuncertainties: np.ndarray = np.asarray(power_unc, dtype=np.float64)
        if len(self.yuncertainties) != len(self.yvalues):
            raise ValueError(
                f"power_unc length ({len(self.yuncertainties)}) must match "
                f"power length ({len(self.yvalues)})"
            )

    def copy(self) -> "PowerMeasurement":
        return PowerMeasurement(
            np.copy(self.yvalues), np.copy(self.yuncertainties),
            self.xstart, self.xstop,
            metadata=copy.deepcopy(self.metadata),
        )

    def subspectrum(self, starti: int, endi: int) -> "PowerMeasurement":
        return PowerMeasurement(
            self.yvalues[starti:endi],
            self.yuncertainties[starti:endi],
            self.get_x_at_index(starti),
            self.get_x_at_index(endi),
            metadata=copy.deepcopy(self.metadata),
        )

    def _new_like(self, new_yvalues: np.ndarray) -> "PowerMeasurement":
        # When arithmetic changes yvalues, scale uncertainties by the same
        # element-wise ratio (valid for multiplicative ops; additive ops
        # would need quadrature addition, but the base class __add__ is
        # used for combining series of the same type which share a scale).
        ratio = np.where(
            self.yvalues != 0,
            np.abs(new_yvalues / np.where(self.yvalues != 0, self.yvalues, 1.0)),
            1.0,
        )
        return PowerMeasurement(
            new_yvalues, self.yuncertainties * ratio,
            self.xstart, self.xstop,
            metadata=copy.deepcopy(self.metadata),
        )

    # ---- weighted combination ------------------------------------------
    def update_bin_with_additional_measurement(
        self, binno: int, p: float, dp: float
    ) -> None:
        """Inverse-variance update of a single bin."""
        if dp <= 0:
            raise ValueError(f"Uncertainty dp must be positive, got {dp}")
        p1 = self.yvalues[binno]
        w1 = 1.0 / (self.yuncertainties[binno] ** 2)
        w2 = 1.0 / (dp ** 2)
        self.yvalues[binno] = (p1 * w1 + p * w2) / (w1 + w2)
        self.yuncertainties[binno] = math.sqrt(1.0 / (w1 + w2))

    def update_with_additional_measurement(self, other: "PowerMeasurement") -> None:
        """
        In-place inverse-variance merge with *other* over the overlapping
        frequency range.  Half-bin misalignment is tolerated.
        """
        i_start = max(0, self.get_x_index_below_x(other.xstart))
        i_stop = min(len(self), self.get_x_index_below_x(other.xstop))
        for i in range(i_start, i_stop):
            x = self.get_x_at_index(i)
            j = other.get_x_index_below_x(x)
            self.update_bin_with_additional_measurement(
                i, other.yvalues[j], other.yuncertainties[j]
            )

    def __str__(self) -> str:
        lines = [f'# "{self.xunits}" "{self.yunits}" "uncertainty"']
        for x, y, dy in zip(self.get_xvalues(), self.yvalues, self.yuncertainties):
            lines.append(f"{x} {y} {dy}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ComplexVoltageSeries  –  complex voltage FFT amplitudes vs frequency
# ---------------------------------------------------------------------------
class ComplexVoltageSeries(ADMXDataSeries):


    def __init__(
        self,
        yvalues,
        f_baseband_start_hz: float,
        f_baseband_stop_hz: float,
        f_abs_start_hz: float,
        scan_params: Optional[ScanParameters] = None,
        delta_f_hz: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        super().__init__(
            np.asarray(yvalues, dtype=np.complex128),
            f_baseband_start_hz,
            f_baseband_stop_hz,
            xunits="Baseband frequency (Hz)",
            yunits="Voltage (V)",
            metadata=metadata,
        )
        self.f_abs_start_hz = float(f_abs_start_hz)
        self.scan_params = scan_params

        # Store explicit delta_f or compute it
        if delta_f_hz is not None:
            self._delta_f_hz = float(delta_f_hz)
        else:
            self._delta_f_hz = self.get_xspacing()

    @property
    def delta_f_hz(self) -> float:
        return self._delta_f_hz

    @property
    def f_abs_hz(self) -> np.ndarray:

        return self.f_abs_start_hz + self.get_xvalues()

    @property
    def f_baseband_hz(self) -> np.ndarray:
        return self.get_xvalues()

    @property
    def power_spectrum(self) -> PowerSpectrum:
        return PowerSpectrum(
            np.abs(self.yvalues) ** 2,
            self.xstart, self.xstop,
            metadata=copy.deepcopy(self.metadata),
        )

    def copy(self) -> "ComplexVoltageSeries":
        return ComplexVoltageSeries(
            np.copy(self.yvalues),
            self.xstart, self.xstop,
            self.f_abs_start_hz,
            scan_params=copy.deepcopy(self.scan_params),
            delta_f_hz=self._delta_f_hz,
            metadata=copy.deepcopy(self.metadata),
        )

    def _new_like(self, new_yvalues: np.ndarray) -> "ComplexVoltageSeries":
        return ComplexVoltageSeries(
            new_yvalues,
            self.xstart, self.xstop,
            self.f_abs_start_hz,
            scan_params=self.scan_params,
            delta_f_hz=self._delta_f_hz,
            metadata=copy.deepcopy(self.metadata),
        )

    def __repr__(self) -> str:
        abs_start = self.f_abs_start_hz / 1e6
        abs_stop = (self.f_abs_start_hz + self.xstop) / 1e6
        return (
            f"ComplexVoltageSeries(N={len(self)}, "
            f"f_abs=[{abs_start:.3f}, {abs_stop:.3f}] MHz, "
            f"df={self._delta_f_hz:.4f} Hz)"
        )