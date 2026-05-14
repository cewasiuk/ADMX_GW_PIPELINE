# """
# create_waveform_template.py
# ---------------------------
# Build a frequency-domain GW voltage template for matched filtering against
# ADMX receiver-flattened data.

# Physics summary
# ---------------
# Berlin et al. (2021), Eq. 23 gives the signal *power* deposited in the
# cavity by a GW of strain h₀:

#     P_sig = ½ Q ω_g³ V_cav^(5/3) (η_n h₀ B₀)²          [Watts]

# where ω_g = 2π f_g is the **angular** GW frequency [rad/s].

# The corresponding voltage amplitude across a Z₀ = 50 Ω transmission line is:

#     |V_sig(f)| = √(2 Z₀ P_sig)
#                = η_n h₀ B₀ √( Q Z₀ ω_g³ V_cav^(5/3) )   [V·s^{1/2}]

# For a chirp signal h̃(f) = |h̃(f)| e^{iΦ(f)}, the complex voltage template is:

#     Ṽ_template(f) = |V_sig(f)| / h₀ × h̃(f)
#                   = scale(f) × h̃(f)

# where scale(f) = η_n B₀ √( Q Z₀ (2πf)³ V_cav^(5/3) ).

# The cavity Lorentzian H_cav(f) is then applied to both the template and the
# data (via hr_raw_prep.py, which does NOT divide it out).  It cancels in the
# SNR but must be present in both to match units.

# Refer to Run1b Reanalysis Paper for full theoretical summary
# """

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pycbc.types import FrequencySeries
from pycbc.waveform import get_fd_waveform
from scipy.interpolate import PchipInterpolator

from admx_db_datatypes import ComplexVoltageSeries, ScanParameters
from config_file_handling import RunConfig
from hr_raw_prep import cavity_lorentzian, load_prepared_scan

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU0   = 4.0 * np.pi * 1e-7   # vacuum permeability [H/m]
C_LIGHT = 299_792_458.0       # speed of light [m/s]
Z0    = 50.0                   # RF impedance [Ω]
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_py(v):
    """Decode numpy scalar or bytes to native Python type."""
    if isinstance(v, (np.bytes_, bytes)):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def _interp_complex_amp_phase(
    wf_fd,
    f_native: np.ndarray,
    f_target: np.ndarray,
    amp_floor: float = 0.0,
    fill_value: complex = 0.0,
) -> np.ndarray:
    y = wf_fd.numpy() if hasattr(wf_fd, "numpy") else np.asarray(wf_fd)
    y = np.asarray(y, dtype=np.complex128)
    f_native = np.asarray(f_native, dtype=np.float64)
    f_target = np.asarray(f_target, dtype=np.float64)

    if y.ndim != 1 or f_native.ndim != 1:
        raise ValueError("wf_fd and f_native must be 1-D")
    if len(y) != len(f_native):
        raise ValueError("len(wf_fd) must equal len(f_native)")

    # sort ascending
    s = np.argsort(f_native)
    f = f_native[s]
    y = y[s]

    amp   = np.abs(y)
    phase = np.unwrap(np.angle(y))

    mask = (amp > amp_floor) & np.isfinite(amp) & np.isfinite(phase)
    if np.count_nonzero(mask) < 4:
        # fall back to real/imag linear interpolation
        re = np.interp(f_target, f, y.real, left=0.0, right=0.0)
        im = np.interp(f_target, f, y.imag, left=0.0, right=0.0)
        return re + 1j * im

    f_m, amp_m, ph_m = f[mask], amp[mask], phase[mask]

    # PCHIP for amplitude (monotone, no overshoot)
    amp_itp = PchipInterpolator(f_m, amp_m, extrapolate=False)
    A  = amp_itp(f_target)
    # linear for phase (unwrapped → linear is fine)
    PH = np.interp(f_target, f_m, ph_m, left=np.nan, right=np.nan)

    out = np.full(len(f_target), fill_value, dtype=np.complex128)
    ok  = np.isfinite(A) & np.isfinite(PH)
    out[ok] = A[ok] * np.exp(1j * PH[ok])
    return out


# ---------------------------------------------------------------------------
# Voltage scale factor  (Berlin Eq. 23, corrected for ω = 2πf)
# ---------------------------------------------------------------------------

def berlin_voltage_scale(
    f_abs_hz: np.ndarray,
    scan_params: ScanParameters,
) -> np.ndarray:
    f = np.asarray(f_abs_hz, dtype=np.float64)

    # avoid zero at DC
    f_safe = f.copy()
    if len(f_safe) > 1:
        f_safe[0] = f_safe[1]
    elif len(f_safe) == 1:
        f_safe[0] = 1.0

    omega = TWO_PI * f_safe                    
    MU0 = 4.0 * np.pi * 1e-7
    C_LIGHT = 299_792_458.0

    scale = np.sqrt(
        (1.0 / (MU0 * C_LIGHT**2))
        * 0.5
        * scan_params.quality_factor
        * (omega ** 3)
        * (scan_params.volume_m3 ** (5.0 / 3.0))
    )
    scale *= scan_params.eta * scan_params.b_field_tesla

    if len(scale) > 0:
        scale[0] = 0.0  # zero DC component

    return scale


# ---------------------------------------------------------------------------
# Cavity Lorentzian transfer function (re-exported for convenience)
# ---------------------------------------------------------------------------

def cavity_lorentzian_on_baseband(
    f_baseband_hz: np.ndarray,
    scan_params: ScanParameters,
    complex_response: bool = True,
):
    f_abs = f_baseband_hz + scan_params.start_freq_hz
    H = cavity_lorentzian(
        f_abs_hz=f_abs,
        f0_hz=scan_params.f0_hz,
        Q=scan_params.quality_factor,
        beta=0.1, ### PLACEHOLDER FOR COUPLING, RELIANT ON SEPHORA
    )
    f0_bb = scan_params.f0_hz - scan_params.start_freq_hz
    if not complex_response:
        return np.abs(H), f0_bb
    return H, f0_bb


# ---------------------------------------------------------------------------
# Template builder
# ---------------------------------------------------------------------------

def build_voltage_template(
    scan_params: ScanParameters,
    f_baseband_hz: np.ndarray,
    f_abs_hz: np.ndarray,
    delta_f_hz: float,
    approximant: str = "TaylorF2",
    mass1: float = 1.4,
    mass2: float = 1.4,
    f_lower_override: Optional[float] = None,
    amp_floor: float = 0.0,
    apply_cavity: bool = True,
    complex_cavity: bool = True,
) -> dict:
    N = len(f_baseband_hz)

    # ------------------------------------------------------------------
    # 1. Generate pyCBC strain waveform - Dont really use this anymore, but included for binary mergers
    # ------------------------------------------------------------------
    f_lower = f_lower_override if f_lower_override is not None else max(
        1.0, float(f_baseband_hz[1]) if N > 1 else 1.0
    )

    try:
        hp, _ = get_fd_waveform(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            delta_f=delta_f_hz,
            f_lower=f_lower,
        )
    except Exception as exc:
        raise RuntimeError(
            f"pyCBC waveform generation failed "
            f"(approximant={approximant}, m1={mass1}, m2={mass2}, "
            f"delta_f={delta_f_hz:.4f}, f_lower={f_lower:.4f}): {exc}"
        ) from exc

    f_hp = np.arange(len(hp), dtype=np.float64) * float(hp.delta_f)

    # ------------------------------------------------------------------
    # 2. Interpolate strain onto baseband grid
    # ------------------------------------------------------------------
    H_strain = _interp_complex_amp_phase(
        hp, f_hp, f_baseband_hz,
        amp_floor=amp_floor, fill_value=0.0,
    )

    # ------------------------------------------------------------------
    # 3. Berlin voltage scale  (uses angular frequency ω = 2πf)
    # ------------------------------------------------------------------
    scale = berlin_voltage_scale(f_abs_hz, scan_params)

    V_no_cavity = scale * H_strain
    V_no_cavity[0] = 0.0 + 0.0j  # zero DC

    # ------------------------------------------------------------------
    # 4. Cavity Lorentzian
    # ------------------------------------------------------------------
    H_cav = None
    f0_bb = None
    V_template = V_no_cavity.copy()

    if apply_cavity:
        H_cav, f0_bb = cavity_lorentzian_on_baseband(
            f_baseband_hz, scan_params, complex_response=complex_cavity
        )
        V_template = V_no_cavity * H_cav
        V_template[0] = 0.0 + 0.0j

    # ------------------------------------------------------------------
    # 5. Wrap as pyCBC FrequencySeries for matched_filter_core
    # ------------------------------------------------------------------
    pycbc_template = FrequencySeries(
        V_template.astype(np.complex128),
        delta_f=delta_f_hz,
    )

    return {
        "V_template":     V_template,
        "V_no_cavity":    V_no_cavity,
        "scale":          scale,
        "H_cav":          H_cav,
        "H_strain":       H_strain,
        "f_baseband_hz":  f_baseband_hz,
        "f_abs_hz":       f_abs_hz,
        "delta_f_hz":     delta_f_hz,
        "f0_baseband_hz": f0_bb,
        "pycbc_template": pycbc_template,
        "scan_params":    scan_params,
        "approximant":    approximant,
        "mass1":          mass1,
        "mass2":          mass2,
    }

# ---------------------------------------------------------------------------
# Convenience: build template from a prepared H5 file
# ---------------------------------------------------------------------------

def build_template_for_scan(
    prepared_h5: Path,
    approximant: str = "TaylorF2",
    mass1: float = 1.4,
    mass2: float = 1.4,
    f_lower_override: Optional[float] = None,
    amp_floor: float = 0.0,
    apply_cavity: bool = True,
    complex_cavity: bool = True,
) -> Optional[dict]:
    result = load_prepared_scan(prepared_h5)
    if result is None:
        log.debug("build_template_for_scan: %s is cut or missing", prepared_h5.name)
        return None

    sp: ScanParameters = result["scan_params"]
    if sp is None:
        log.warning("build_template_for_scan: no ScanParameters in %s", prepared_h5.name)
        return None

    tmpl = build_voltage_template(
        scan_params=sp,
        f_baseband_hz=result["f_baseband"],
        f_abs_hz=result["f_abs"],
        delta_f_hz=result["delta_f_hz"],
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        f_lower_override=f_lower_override,
        amp_floor=amp_floor,
        apply_cavity=apply_cavity,
        complex_cavity=complex_cavity,
    )

    tmpl["data"]    = result["cvs"]
    tmpl["V_data"]  = result["V_flat"]
    tmpl["PSD"]     = result["PSD"]
    return tmpl


# ---------------------------------------------------------------------------
# Template bank over a mass grid
# ---------------------------------------------------------------------------

def build_template_bank(
    scan_params: ScanParameters,
    f_baseband_hz: np.ndarray,
    f_abs_hz: np.ndarray,
    delta_f_hz: float,
    mass_grid: list[tuple[float, float]],
    approximant: str = "TaylorF2",
    apply_cavity: bool = True,
    complex_cavity: bool = True,
) -> list[dict]:

    bank = []
    for m1, m2 in mass_grid:
        try:
            tmpl = build_voltage_template(
                scan_params=scan_params,
                f_baseband_hz=f_baseband_hz,
                f_abs_hz=f_abs_hz,
                delta_f_hz=delta_f_hz,
                approximant=approximant,
                mass1=m1,
                mass2=m2,
                apply_cavity=apply_cavity,
                complex_cavity=complex_cavity,
            )
            tmpl["mass_pair"] = (m1, m2)
            bank.append(tmpl)
        except Exception as exc:
            log.warning(
                "build_template_bank: failed for (%.2f, %.2f) Msun: %s",
                m1, m2, exc,
            )
    log.info("Built template bank: %d / %d templates", len(bank), len(mass_grid))
    return bank

# ---------------------------------------------------------------------------
# This is the current template bank used in the analysis, built from lorentzian ( frequency domain ) for Boson Annihilation 
# ---------------------------------------------------------------------------

def build_superradiant_template(
    scan_params: ScanParameters,
    f_baseband_hz: np.ndarray,
    delta_f_hz: float,
    f_signal_offset_hz: float = 0.0,
) -> dict:
    MU0     = 4.0 * np.pi * 1e-7
    C_LIGHT = 299_792_458.0

    sp     = scan_params
    f0     = sp.f0_hz
    Q      = sp.quality_factor
    omega0 = 2.0 * np.pi * f0

    scale_at_f0 = float(
        np.sqrt(
            (1.0 / (MU0 * C_LIGHT**2))
            * 0.5
            * Q
            * omega0**3
            * sp.volume_m3**(5.0 / 3.0)
        )
        * sp.eta
        * sp.b_field_tesla
    )

    tau_ring_s          = Q / (np.pi * f0)
    linewidth_hz        = f0 / Q
    n_bins_in_linewidth = int(round(linewidth_hz / delta_f_hz))

    f0_baseband_hz = f0 - sp.start_freq_hz


    f_center = f0_baseband_hz + f_signal_offset_hz
    H_cav    = 1.0 / (1.0 + 2j * Q * (f_baseband_hz - f_center) / f0)
    H_cav    = np.asarray(H_cav, dtype=np.complex128)

    V_template      = scale_at_f0 * H_cav
    V_template      = np.asarray(V_template, dtype=np.complex128)
    V_template[0]   = 0.0 + 0.0j

    pycbc_template = FrequencySeries(V_template.copy(), delta_f=delta_f_hz)
    if scale_at_f0 > 0:
        V_normed = V_template / scale_at_f0
    else:
        log.warning("scale_at_f0 = 0 for scan %s — using raw template", sp)
        V_normed = V_template.copy()

    V_normed[0] = 0.0 + 0.0j
    pycbc_template_normed = FrequencySeries(
        np.asarray(V_normed, dtype=np.complex128), delta_f=delta_f_hz
    )

    return {
        'V_template':           V_template,
        'scale_at_f0':          scale_at_f0,
        'H_cav':                H_cav,
        'f0_baseband_hz':       f0_baseband_hz,
        'f_signal_offset_hz':   f_signal_offset_hz,
        'tau_ring_s':           tau_ring_s,
        'linewidth_hz':          linewidth_hz,
        'n_bins_in_linewidth':   n_bins_in_linewidth,
        'pycbc_template':        pycbc_template,         # raw V/strain — diagnostics only
        'pycbc_template_normed': pycbc_template_normed,  # use this in matched_filter
        'scan_params':           scan_params,
    }

# ---------------------------------------------------------------------------
# Strain sensitivity limit from SNR threshold
# ---------------------------------------------------------------------------


def snr_threshold_to_strain(
    rho_threshold: float,
    scan_params: ScanParameters,
    f_signal_hz: float,
) -> float:
    f_arr = np.array([f_signal_hz], dtype=np.float64)
    scale = berlin_voltage_scale(f_arr, scan_params)
    if scale[0] <= 0:
        raise ValueError(f"scale(f={f_signal_hz:.3e} Hz) = 0; check scan_params.")
    return float(rho_threshold / scale[0])


# ---------------------------------------------------------------------------
# Save / load template H5
# ---------------------------------------------------------------------------

def save_template_h5(out_path: Path, tmpl: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sp: ScanParameters = tmpl["scan_params"]

    with h5py.File(out_path, "w") as f:
        f.create_dataset("f_baseband_hz", data=tmpl["f_baseband_hz"].astype(np.float64))
        f.create_dataset("f_abs_hz",      data=tmpl["f_abs_hz"].astype(np.float64))
        f.create_dataset("H_strain",      data=tmpl["H_strain"].astype(np.complex128))
        f.create_dataset("V_no_cavity",   data=tmpl["V_no_cavity"].astype(np.complex128))
        f.create_dataset("V_template",    data=tmpl["V_template"].astype(np.complex128))
        f.create_dataset("scale",         data=tmpl["scale"].astype(np.float64))

        if tmpl["H_cav"] is not None:
            f.create_dataset("H_cav", data=np.asarray(tmpl["H_cav"]).astype(np.complex128))

        f.attrs["delta_f_hz"]     = float(tmpl["delta_f_hz"])
        f.attrs["approximant"]    = tmpl["approximant"]
        f.attrs["mass1_msun"]     = float(tmpl["mass1"])
        f.attrs["mass2_msun"]     = float(tmpl["mass2"])
        f.attrs["apply_cavity"]   = int(tmpl["H_cav"] is not None)
        f.attrs["pipeline_stage"] = "create_waveform_template"

        if tmpl["f0_baseband_hz"] is not None:
            f.attrs["f0_baseband_hz"] = float(tmpl["f0_baseband_hz"])

        f.attrs["units_V_template"] = (
            "Complex voltage template [V/strain × strain = V]. "
            "scale(f) = eta*B0*sqrt(Q*Z0*(2*pi*f)^3*Vcav^(5/3)), "
            "with 2*pi factor included (angular frequency). "
            "Cavity Lorentzian applied. "
            "Use directly with matched_filter_core.py."
        )

        # scan parameters sub-group
        if sp is not None:
            g = f.require_group("scan_params")
            g.attrs["f0_hz"]              = float(sp.f0_hz)
            g.attrs["quality_factor"]     = float(sp.quality_factor)
            g.attrs["b_field_tesla"]      = float(sp.b_field_tesla)
            g.attrs["coupling"]           = float(sp.coupling)
            g.attrs["volume_m3"]          = float(sp.volume_m3)
            g.attrs["eta"]                = float(sp.eta)
            g.attrs["start_freq_hz"]      = float(sp.start_freq_hz)
            g.attrs["stop_freq_hz"]       = float(sp.stop_freq_hz)
            g.attrs["integration_time_s"] = float(sp.integration_time_s)
            g.attrs["tsys_kelvin"]        = float(sp.tsys_kelvin)

    log.info("Saved template → %s", out_path)


def load_template_h5(path: Path) -> dict:
    """Load a saved voltage template H5."""
    with h5py.File(path, "r") as f:
        out = {
            "f_baseband_hz": f["f_baseband_hz"][:].astype(np.float64),
            "f_abs_hz":      f["f_abs_hz"][:].astype(np.float64),
            "H_strain":      f["H_strain"][:].astype(np.complex128),
            "V_no_cavity":   f["V_no_cavity"][:].astype(np.complex128),
            "V_template":    f["V_template"][:].astype(np.complex128),
            "scale":         f["scale"][:].astype(np.float64),
            "H_cav":         f["H_cav"][:].astype(np.complex128) if "H_cav" in f else None,
            "delta_f_hz":    float(f.attrs["delta_f_hz"]),
            "approximant":   str(f.attrs.get("approximant", "unknown")),
            "mass1":         float(f.attrs.get("mass1_msun", 0.0)),
            "mass2":         float(f.attrs.get("mass2_msun", 0.0)),
            "f0_baseband_hz": float(f.attrs["f0_baseband_hz"]) if "f0_baseband_hz" in f.attrs else None,
        }
        out["pycbc_template"] = FrequencySeries(
            out["V_template"].astype(np.complex128),
            delta_f=out["delta_f_hz"],
        )
    return out


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_template(
    tmpl: dict,
    title: str = "",
    show: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """Four-panel diagnostic plot for a voltage template."""
    f_bb  = tmpl["f_baseband_hz"]
    f_khz = f_bb / 1e3
    V     = tmpl["V_template"]
    Vnc   = tmpl["V_no_cavity"]
    sc    = tmpl["scale"]
    H_s   = tmpl["H_strain"]
    sp    = tmpl["scan_params"]
    f0_bb = tmpl.get("f0_baseband_hz")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    label = (
        f"{title}  |  {tmpl['approximant']}  "
        f"m1={tmpl['mass1']:.1f} m2={tmpl['mass2']:.1f} M☉"
    )
    if sp:
        label += f"\nf0={sp.f0_hz/1e6:.3f} MHz  Q={sp.quality_factor:.0f}  B={sp.b_field_tesla:.2f} T"
    fig.suptitle(label, fontsize=11)

    mask = f_bb > 0

    # Panel 1: strain waveform amplitude
    ax = axes[0, 0]
    ax.semilogy(f_khz[mask], np.abs(H_s[mask]) + 1e-50, lw=1)
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("|h̃(f)| [strain]")
    ax.set_title("Strain waveform amplitude")
    ax.grid(True, alpha=0.3)

    # Panel 2: Berlin scale factor
    ax = axes[0, 1]
    ax.semilogy(f_khz[mask], sc[mask], lw=1, color="C1")
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("scale(f) = |V| / strain [V/strain]")
    ax.set_title("Berlin voltage scale  (ω = 2πf corrected)")
    ax.grid(True, alpha=0.3)

    # Panel 3: template amplitude with/without cavity
    ax = axes[1, 0]
    ax.semilogy(f_khz[mask], np.abs(Vnc[mask]) + 1e-50, lw=1, label="no H_cav")
    ax.semilogy(f_khz[mask], np.abs(V[mask])   + 1e-50, lw=1, ls="--", label="with H_cav")
    if f0_bb is not None:
        ax.axvline(f0_bb / 1e3, ls=":", lw=0.8, color="gray", label=f"f₀={f0_bb/1e3:.1f} kHz")
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("|Ṽ_template(f)| [V]")
    ax.set_title("Voltage template amplitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: template phase
    ax = axes[1, 1]
    phase = np.unwrap(np.angle(V))
    ax.plot(f_khz[mask], phase[mask], lw=0.8, color="C3")
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("∠Ṽ_template(f) [rad]")
    ax.set_title("Template phase (unwrapped)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved template plot → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build GW voltage template for ADMX matched filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--prepared_h5", required=True,
        help="Path to a *_prepared.h5 file from hr_raw_prep.py.",
    )
    p.add_argument("--approximant", default="TaylorF2")
    p.add_argument("--mass1", type=float, default=1.4)
    p.add_argument("--mass2", type=float, default=1.4)
    p.add_argument("--f_lower", type=float, default=None,
                   help="Override lower frequency cutoff [Hz].")
    p.add_argument("--no_cavity", action="store_true",
                   help="Do NOT apply cavity Lorentzian (must match hr_raw_prep).")
    p.add_argument("--mag_cavity", action="store_true",
                   help="Use |H_cav| instead of complex H_cav.")
    p.add_argument("--save_h5", default="",
                   help="Save template to this H5 path.")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--save_plot", default="")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    tmpl = build_template_for_scan(
        prepared_h5=Path(args.prepared_h5),
        approximant=args.approximant,
        mass1=args.mass1,
        mass2=args.mass2,
        f_lower_override=args.f_lower,
        apply_cavity=not args.no_cavity,
        complex_cavity=not args.mag_cavity,
    )

    if tmpl is None:
        print("Scan is cut or could not be loaded.")
        return

    sp = tmpl["scan_params"]
    print(f"Template built for: {sp}")
    print(f"  Approximant: {tmpl['approximant']}")
    print(f"  Masses: {tmpl['mass1']:.2f} + {tmpl['mass2']:.2f} M☉")
    print(f"  max|Ṽ_template|: {np.max(np.abs(tmpl['V_template'])):.3e} V")
    print(f"  scale at f0: {np.interp(sp.f0_hz, tmpl['f_abs_hz'], tmpl['scale']):.3e} V/strain")

    if sp:
        h_min = snr_threshold_to_strain(3.0, sp, sp.f0_hz)
        print(f"  3σ strain sensitivity at f0: {h_min:.3e}")

    if args.save_h5:
        save_template_h5(Path(args.save_h5), tmpl)

    if args.plot or args.save_plot:
        plot_template(
            tmpl,
            title=Path(args.prepared_h5).stem,
            show=args.plot,
            save_path=Path(args.save_plot) if args.save_plot else None,
        )


if __name__ == "__main__":
    main()
