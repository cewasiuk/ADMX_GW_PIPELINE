# """
# hr_raw_prep.py
# --------------
# Stage 3 of the ADMX GW pipeline: read receiver-flattened complex voltage
# spectra produced by binned4.py, apply the cavity Lorentzian correction,
# compute a per-scan noise PSD, and write matched-filter-ready HDF5 files.

# Pipeline position
# -----------------
#     binned4.py           →  FFT_BinnedFlat  (dimensionless, receiver-divided)
#     hr_raw_prep.py       →  V_cavity        (Lorentzian-corrected voltage, V·√s)
#     create_waveform_template.py  →  V_template  (same units, from Berlin Eq.23)
#     matched_filter_core.py       →  SNR time series
# """

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.psd import welch
import pyfftw

from admx_db_datatypes import ComplexVoltageSeries, ScanParameters
from config_file_handling import (
    RunConfig,
    find_parameter_row,
    parse_filename_timestamp,
)

log = logging.getLogger(__name__)
pyfftw.interfaces.cache.enable()

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
K_BOLTZ = 1.380_649e-23   # J/K
Z0 = 50.0                  # Ω — standard RF impedance

# ---------------------------------------------------------------------------
# Lorentzian cavity transfer function
# ---------------------------------------------------------------------------

def cavity_lorentzian(
    f_abs_hz: np.ndarray,
    f0_hz: float,
    Q: float,
    beta: float,
) -> np.ndarray:
    if Q <= 0:
        raise ValueError(f"Q must be positive, got {Q}")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")

    coupling_voltage = np.sqrt(beta / (1.0 + beta)) if beta > 0 else 1.0
    x = 2.0 * Q * (f_abs_hz - f0_hz) / f0_hz
    return coupling_voltage / (1.0 + 1j * x)


def expected_noise_stdev(integration_time_s: float, delta_f_hz: float) -> float:
    if integration_time_s <= 0 or delta_f_hz <= 0:
        raise ValueError("integration_time_s and delta_f_hz must be positive")
    return 1.0 / np.sqrt(integration_time_s * delta_f_hz)


# ---------------------------------------------------------------------------
# PSD estimation
# ---------------------------------------------------------------------------

def estimate_psd_welch(
    v_flat: np.ndarray,
    delta_f_hz: float,
    seg_len_factor: int = 4,
) -> np.ndarray:
    N_freq = len(v_flat)

    # reconstruct a real time series via irfft
    # irfft of an N_freq-point rFFT output has 2*(N_freq-1) time samples
    x_time = pyfftw.interfaces.numpy_fft.irfft(v_flat, n=2 * (N_freq - 1))
    N_time = len(x_time)

    # sample spacing in pseudo-time domain
    # The rFFT covers frequencies 0 … f_max = (N_freq-1)*delta_f
    # so delta_t = 1/(2*f_max) = 1/(2*(N_freq-1)*delta_f)
    f_max = (N_freq - 1) * delta_f_hz
    if f_max <= 0:
        raise ValueError("Cannot estimate PSD: f_max <= 0")
    delta_t = 1.0 / (2.0 * f_max)

    ts = TimeSeries(x_time.astype(np.float64), delta_t=delta_t)

    seg_len = max(32, N_time // seg_len_factor)
    # ensure seg_len is even (required by pycbc welch)
    if seg_len % 2 != 0:
        seg_len -= 1

    psd_ts = welch(
        ts,
        seg_len=seg_len,
        seg_stride=seg_len // 2,
        window="hann",
        avg_method="median",
    )

    # interpolate Welch PSD (coarser grid) onto original delta_f grid
    f_welch = np.array(psd_ts.sample_frequencies)
    psd_vals = np.array(psd_ts)

    f_target = np.arange(N_freq, dtype=np.float64) * delta_f_hz
    psd_interp = np.interp(f_target, f_welch, psd_vals,
                           left=psd_vals[0], right=psd_vals[-1])

    # guard against zeros / negatives
    psd_floor = np.finfo(np.float64).tiny
    return np.maximum(psd_interp, psd_floor)


# ---------------------------------------------------------------------------
# Core per-scan processing
# ---------------------------------------------------------------------------

def process_binned_h5(
    binned_h5_path: Path,
    scan_params: ScanParameters,
    max_stdev_ratio: float = 3.0,
    psd_seg_len_factor: int = 4,
) -> Optional[dict]:
    if not binned_h5_path.exists():
        log.warning("process_binned_h5: file not found: %s", binned_h5_path)
        return None

    try:
        with h5py.File(binned_h5_path, "r") as f:
            # Read flat complex FFT
            v_re = f["FFT_BinnedFlat_Re"][:].astype(np.float64) if "FFT_BinnedFlat_Re" in f else None
            v_im = f["FFT_BinnedFlat_Im"][:].astype(np.float64) if "FFT_BinnedFlat_Im" in f else None

            # Prefer unified complex dataset (new format from updated binned4)
            if "FFT_BinnedFlat" in f:
                v_flat = f["FFT_BinnedFlat"][:].astype(np.complex128)
            elif v_re is not None and v_im is not None:
                v_flat = v_re + 1j * v_im
            else:
                raise KeyError("No FFT_BinnedFlat dataset found in H5.")

            f_baseband = f["f_baseband"][:].astype(np.float64)
            f_abs = f["f_abs"][:].astype(np.float64)

            delta_f = float(f.attrs.get("delta_f_out",
                            f.attrs.get("delta_f_binned",
                            (f_baseband[1] - f_baseband[0]) if len(f_baseband) > 1 else 1.0)))
            fstart_abs = float(f.attrs.get("fstart_abs_hz", f_abs[0]))
            fstop_abs = float(f.attrs.get("fstop_abs_hz", f_abs[-1]))

    except Exception as exc:
        log.error("process_binned_h5: failed to read %s: %s", binned_h5_path, exc)
        return None

    if len(v_flat) != len(f_baseband):
        log.error(
            "process_binned_h5: length mismatch v_flat=%d f_baseband=%d in %s",
            len(v_flat), len(f_baseband), binned_h5_path,
        )
        return None

    # ------------------------------------------------------------------
    # Data quality: σ-excess cut
    # ------------------------------------------------------------------
    power = np.abs(v_flat) ** 2
    measured_std = float(np.std(power))
    measured_mean = float(np.mean(power))

    # Normalise to unit mean before computing σ ratio
    if measured_mean > 0:
        norm_std = measured_std / measured_mean
    else:
        norm_std = float("inf")

    try:
        expected_std = expected_noise_stdev(
            scan_params.integration_time_s, delta_f
        )
    except ValueError:
        expected_std = float("nan")

    stdev_ratio = norm_std / expected_std if np.isfinite(expected_std) and expected_std > 0 else float("nan")

    cut_reason = ""
    if np.isfinite(stdev_ratio) and stdev_ratio > max_stdev_ratio:
        cut_reason = (
            f"stdev_excess: ratio={stdev_ratio:.2f} > {max_stdev_ratio:.1f}. "
            "Possible RFI or JPA instability."
        )
        log.info("Cut scan %s: %s", binned_h5_path.name, cut_reason)

    # ------------------------------------------------------------------
    # PSD estimate — one-sided, consistent with matched-filter inner product
    # S_n(f) = 2|V_flat|²/Δf for interior bins (DC/Nyquist not doubled)
    # ------------------------------------------------------------------
    power = np.abs(v_flat) ** 2
    psd   = 2.0 * power / delta_f
    psd[0]  = power[0]  / delta_f
    psd[-1] = power[-1] / delta_f
    psd   = np.maximum(psd, np.finfo(np.float64).tiny)

    # ------------------------------------------------------------------
    # Package as ComplexVoltageSeries
    # ------------------------------------------------------------------
    f_bb_start = float(f_baseband[0])
    f_bb_stop = float(f_baseband[-1]) + delta_f

    cvs = ComplexVoltageSeries(
        yvalues=v_flat,
        f_baseband_start_hz=f_bb_start,
        f_baseband_stop_hz=f_bb_stop,
        f_abs_start_hz=fstart_abs,
        scan_params=scan_params,
        delta_f_hz=delta_f,
        metadata={
            "source_h5": str(binned_h5_path),
            "stdev_ratio": stdev_ratio,
            "cut_reason": cut_reason,
            "pipeline_stage": "hr_raw_prep",
        },
    )

    return {
        "cvs": cvs,
        "psd": psd,
        "f_baseband": f_baseband,
        "f_abs": f_abs,
        "stdev_ratio": stdev_ratio,
        "cut_reason": cut_reason,
    }


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------

def save_prepared_scan(
    result: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cvs: ComplexVoltageSeries = result["cvs"]
    psd: np.ndarray = result["psd"]
    sp: ScanParameters = cvs.scan_params

    with h5py.File(output_path, "w") as f:
        # Complex voltage (split for HDF5 portability)
        f.create_dataset("V_flat_re", data=cvs.yvalues.real.astype(np.float64))
        f.create_dataset("V_flat_im", data=cvs.yvalues.imag.astype(np.float64))
        f.create_dataset("PSD",       data=psd.astype(np.float64))
        f.create_dataset("f_baseband", data=result["f_baseband"].astype(np.float64))
        f.create_dataset("f_abs",      data=result["f_abs"].astype(np.float64))

        # File-level attributes
        f.attrs["delta_f_hz"]     = float(cvs.delta_f_hz)
        f.attrs["fstart_abs_hz"]  = float(cvs.f_abs_start_hz)
        f.attrs["fstop_abs_hz"]   = float(cvs.f_abs_start_hz + cvs.xstop)
        f.attrs["cut_reason"]     = result["cut_reason"]
        f.attrs["stdev_ratio"]    = float(result["stdev_ratio"]) if np.isfinite(result["stdev_ratio"]) else -1.0
        f.attrs["pipeline_stage"] = "hr_raw_prep"

        f.attrs["units_V_flat"] = (
            "Receiver-normalised complex FFT amplitude. "
            "Lorentzian NOT divided out — template carries same H_cav factor. "
            "For strain conversion use create_waveform_template.py."
        )
        f.attrs["units_PSD"] = (
            "One-sided Welch PSD of |V_flat|^2 [receiver_units^2 / Hz]. "
            "Use directly for matched-filter whitening (same units as V_flat^2)."
        )

        # ScanParameters as a sub-group for easy downstream access
        if sp is not None:
            g = f.require_group("scan_params")
            g.attrs["f0_hz"]              = float(sp.f0_hz)
            g.attrs["f0_mhz"]             = float(sp.f0_hz / 1e6)
            g.attrs["start_freq_hz"]      = float(sp.start_freq_hz)
            g.attrs["stop_freq_hz"]       = float(sp.stop_freq_hz)
            g.attrs["quality_factor"]     = float(sp.quality_factor)
            g.attrs["b_field_tesla"]      = float(sp.b_field_tesla)
            g.attrs["coupling"]           = float(sp.coupling)
            g.attrs["jpa_snri_db"]        = float(sp.jpa_snri_db)
            g.attrs["thfet_kelvin"]       = float(sp.thfet_kelvin)
            g.attrs["attenuation_db"]     = float(sp.attenuation_db)
            g.attrs["integration_time_s"] = float(sp.integration_time_s)
            g.attrs["volume_m3"]          = float(sp.volume_m3)
            g.attrs["eta"]                = float(sp.eta)
            g.attrs["tsys_kelvin"]        = float(sp.tsys_kelvin)
            g.attrs["timestamp"]          = str(sp.timestamp)
            g.attrs["filename_tag"]       = str(sp.filename_tag)
            g.attrs["cavity_linewidth_hz"] = float(sp.cavity_linewidth_hz)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_hr_raw_prep(
    run_definition_path: str = "run1b_definitions.yaml",
    nibble_name: str = "nibble5",
    param_date: str = "2018_05_19",
    binned_dir: str = "hr_data/binned_hr_data",
    output_dir: str = "hr_data/prepared",
    max_stdev_ratio: float = 3.0,
    psd_seg_len_factor: int = 4,
    overwrite: bool = False,
) -> list[Path]:
    cfg = RunConfig.from_yaml(run_definition_path)
    nibble_cfg = cfg.nibble(nibble_name)
    b_field = cfg.b_field(nibble_name)
    param_df = cfg.load_parameter_df(nibble_name, param_date)

    binned_path = Path(binned_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    binned_files = sorted(binned_path.glob("*_binned.h5"))
    if not binned_files:
        raise FileNotFoundError(
            f"No *_binned.h5 files found in {binned_dir}. "
            "Run binned4.py first."
        )

    log.info(
        "hr_raw_prep: nibble=%s, %d binned files, max_stdev_ratio=%.1f",
        nibble_name, len(binned_files), max_stdev_ratio,
    )

    outputs: list[Path] = []
    n_cut = 0

    for binned_h5 in binned_files:
        stem = binned_h5.stem.replace("_binned", "")
        prepared_path = out_path / f"{stem}_prepared.h5"

        if prepared_path.exists() and not overwrite:
            log.debug("Skipping %s (output exists)", binned_h5.name)
            outputs.append(prepared_path)
            continue

        # --- match parameter row ---
        # binned filename: admx_data_YYYY_MM_DD_HH_MM_SS_channel_1_binned.h5
        # parameter tag:   admx_data_YYYY_MM_DD_HH_MM_SS_channel_1
        dat_name = stem + ".dat"
        row = find_parameter_row(param_df, dat_name)
        if row is None:
            log.info("hr_raw_prep: no parameter row for %s, skipping", binned_h5.name)
            continue

        # --- timestamp cut check ---
        ts = parse_filename_timestamp(binned_h5.name)
        if ts is not None:
            cut, reason = cfg.is_scan_cut(nibble_name, ts)
            if cut:
                log.info("hr_raw_prep: timestamp cut for %s: %s", binned_h5.name, reason)
                n_cut += 1
                # still write a record so downstream knows this file was seen
                scan_params = ScanParameters.from_dataframe_row(row, b_field)
                scan_params.timestamp = ts.isoformat()
                result = {
                    "cvs": ComplexVoltageSeries(
                        yvalues=np.zeros(1, dtype=np.complex128),
                        f_baseband_start_hz=0.0,
                        f_baseband_stop_hz=1.0,
                        f_abs_start_hz=float(row["Start_Frequency"]) * 1e6,
                        scan_params=scan_params,
                    ),
                    "psd": np.ones(1),
                    "f_baseband": np.zeros(1),
                    "f_abs": np.array([float(row["Start_Frequency"]) * 1e6]),
                    "stdev_ratio": float("nan"),
                    "cut_reason": reason,
                }
                save_prepared_scan(result, prepared_path)
                outputs.append(prepared_path)
                continue

        # --- build ScanParameters ---
        scan_params = ScanParameters.from_dataframe_row(row, b_field)
        if ts is not None:
            scan_params.timestamp = ts.isoformat()

        # --- process ---
        result = process_binned_h5(
            binned_h5_path=binned_h5,
            scan_params=scan_params,
            max_stdev_ratio=max_stdev_ratio,
            psd_seg_len_factor=psd_seg_len_factor,
        )

        if result is None:
            log.warning("hr_raw_prep: process_binned_h5 returned None for %s", binned_h5.name)
            continue

        if result["cut_reason"]:
            n_cut += 1

        save_prepared_scan(result, prepared_path)
        outputs.append(prepared_path)
        log.info(
            "Saved %s  (σ_ratio=%.2f%s)",
            prepared_path.name,
            result["stdev_ratio"] if np.isfinite(result["stdev_ratio"]) else -1,
            f", CUT: {result['cut_reason']}" if result["cut_reason"] else "",
        )

    log.info(
        "hr_raw_prep done: %d / %d files written, %d cut",
        len(outputs), len(binned_files), n_cut,
    )
    return outputs


# ---------------------------------------------------------------------------
# Convenience loader for downstream stages
# ---------------------------------------------------------------------------

def load_prepared_scan(prepared_h5_path: Path) -> Optional[dict]:
    if not prepared_h5_path.exists():
        log.warning("load_prepared_scan: not found: %s", prepared_h5_path)
        return None

    with h5py.File(prepared_h5_path, "r") as f:
        cut_reason = str(f.attrs.get("cut_reason", ""))
        if cut_reason:
            log.debug("load_prepared_scan: %s is cut (%s)", prepared_h5_path.name, cut_reason)
            return None

        v_re = f["V_flat_re"][:].astype(np.float64)
        v_im = f["V_flat_im"][:].astype(np.float64)
        v_flat = v_re + 1j * v_im

        psd        = f["PSD"][:].astype(np.float64)
        f_baseband = f["f_baseband"][:].astype(np.float64)
        f_abs      = f["f_abs"][:].astype(np.float64)
        delta_f    = float(f.attrs["delta_f_hz"])
        stdev_ratio = float(f.attrs.get("stdev_ratio", float("nan")))
        fstart_abs  = float(f.attrs["fstart_abs_hz"])

        sp = None
        if "scan_params" in f:
            g = f["scan_params"]
            sp = ScanParameters(
                f0_hz             = float(g.attrs.get("f0_hz", 0.0)),
                start_freq_hz     = float(g.attrs.get("start_freq_hz", 0.0)),
                stop_freq_hz      = float(g.attrs.get("stop_freq_hz", 0.0)),
                quality_factor    = float(g.attrs.get("quality_factor", 1.0)),
                b_field_tesla     = float(g.attrs.get("b_field_tesla", 0.0)),
                coupling          = float(g.attrs.get("coupling", 0.0)),
                jpa_snri_db       = float(g.attrs.get("jpa_snri_db", 0.0)),
                thfet_kelvin      = float(g.attrs.get("thfet_kelvin", 0.0)),
                attenuation_db    = float(g.attrs.get("attenuation_db", 0.0)),
                integration_time_s= float(g.attrs.get("integration_time_s", 0.0)),
                volume_m3         = float(g.attrs.get("volume_m3", 0.136)),
                eta               = float(g.attrs.get("eta", 0.1)),
                timestamp         = str(g.attrs.get("timestamp", "")),
                filename_tag      = str(g.attrs.get("filename_tag", "")),
            )

    cvs = ComplexVoltageSeries(
        yvalues=v_flat,
        f_baseband_start_hz=float(f_baseband[0]),
        f_baseband_stop_hz=float(f_baseband[-1]) + delta_f,
        f_abs_start_hz=fstart_abs,
        scan_params=sp,
        delta_f_hz=delta_f,
    )

    return {
        "cvs":         cvs,
        "V_flat":      v_flat,
        "PSD":         psd,
        "f_baseband":  f_baseband,
        "f_abs":       f_abs,
        "delta_f_hz":  delta_f,
        "scan_params": sp,
        "cut_reason":  cut_reason,
        "stdev_ratio": stdev_ratio,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare binned ADMX scans for matched filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--run_definition", default="run1b_definitions.yaml")
    p.add_argument("-n", "--nibble_name", default="nibble5")
    p.add_argument("--param_date", default="2018_05_19")
    p.add_argument("--binned_dir", default="hr_data/binned_hr_data")
    p.add_argument("--output_dir", default="hr_data/prepared")
    p.add_argument(
        "--max_stdev_ratio", type=float, default=3.0,
        help="σ-excess cut: scans with measured/expected σ above this are flagged.",
    )
    p.add_argument(
        "--psd_seg_len_factor", type=int, default=4,
        help="Welch segment length = N_bins // this value.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    outputs = run_hr_raw_prep(
        run_definition_path=args.run_definition,
        nibble_name=args.nibble_name,
        param_date=args.param_date,
        binned_dir=args.binned_dir,
        output_dir=args.output_dir,
        max_stdev_ratio=args.max_stdev_ratio,
        psd_seg_len_factor=args.psd_seg_len_factor,
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(outputs)} scans → {args.output_dir}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Example CLI usage
# ---------------------------------------------------------------------------
# Standard run for nibble5:
#   python hr_raw_prep.py -n nibble5 --binned_dir hr_data/binned_hr_data
#
# Stricter noise cut (only keep very clean scans):
#   python hr_raw_prep.py -n nibble5 --max_stdev_ratio 2.0
#
# All nibbles:
#   for n in nibble1a nibble1b nibble2 nibble3 nibble4 nibble5; do
#       python hr_raw_prep.py -n $n
#   done