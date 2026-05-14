# """
# psd_computation.py
# ------------------
# Noise PSD estimation for ADMX matched filtering.

# Pipeline position
# -----------------
#     hr_raw_prep.py          ->  *_prepared.h5  (V_flat per scan)
#     psd_computation.py      ->  nibble_psd.h5  (stacked PSD)
#     matched_filter_core.py  ->  SNR time series

# The cross-scan median of |V_flat|^2 is used instead of E[|V_flat|^2] from a
# single scan, making the estimate robust to RFI spikes and mode crossings.

# """

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from pycbc.types import FrequencySeries

from hr_raw_prep import load_prepared_scan
from config_file_handling import RunConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One-sided PSD from raw |V_flat|^2
# ---------------------------------------------------------------------------

def onesided_psd_from_power(
    power: np.ndarray,
    delta_f_hz: float,
) -> np.ndarray:
    df = float(delta_f_hz)
    N = len(power)

    # Interior bins: factor of 2 for one-sided convention
    psd = 2.0 * df * np.asarray(power, dtype=np.float64)

    # DC bin is not doubled
    psd[0] = df * float(power[0])

    # Nyquist bin is not doubled
    if N > 1:
        psd[-1] = df * float(power[-1])

    return np.maximum(psd, np.finfo(np.float64).tiny)


# ---------------------------------------------------------------------------
# Cross-scan PSD stacking
# ---------------------------------------------------------------------------

def compute_psd_from_prepared_dir(
    prepared_dir: str | Path,
    stack_method: str = "median",
    f_low_hz: Optional[float] = None,
    f_high_hz: Optional[float] = None,
    save_per_scan: bool = False,
) -> tuple[FrequencySeries, np.ndarray, np.ndarray, dict]:
    prepared_dir = Path(prepared_dir)
    files = sorted(prepared_dir.glob("*_prepared.h5"))

    if not files:
        raise FileNotFoundError(
            f"No *_prepared.h5 files found in {prepared_dir}. "
            "Run hr_raw_prep.py first."
        )

    power_rows: list[np.ndarray] = []
    skipped: list[str] = []
    ref_N: Optional[int] = None
    ref_df: Optional[float] = None
    f_baseband_ref: Optional[np.ndarray] = None
    f_abs_ref: Optional[np.ndarray] = None

    for path in files:
        result = load_prepared_scan(path)
        if result is None:
            skipped.append(path.name)
            continue

        v_flat  = result["V_flat"]
        f_bb    = result["f_baseband"]
        f_abs   = result["f_abs"]
        delta_f = float(result["delta_f_hz"])
        N       = len(v_flat)

        if ref_N is None:
            ref_N          = N
            ref_df         = delta_f
            f_baseband_ref = f_bb
            f_abs_ref      = f_abs
        else:
            if N != ref_N:
                log.warning("PSD: skipping %s — length %d vs ref %d", path.name, N, ref_N)
                skipped.append(path.name)
                continue
            if not np.isclose(delta_f, ref_df, rtol=1e-6):
                log.warning("PSD: skipping %s — delta_f %.6f vs ref %.6f",
                            path.name, delta_f, ref_df)
                skipped.append(path.name)
                continue

        power_rows.append(np.abs(v_flat) ** 2)

    if not power_rows:
        raise RuntimeError(
            f"No usable scans in {prepared_dir}. "
            f"Skipped {len(skipped)} files. Check hr_raw_prep cuts."
        )

    power_matrix = np.vstack(power_rows)   # (N_scans, N_freq)

    if stack_method == "median":
        stacked_power = np.median(power_matrix, axis=0)
        stacked_power = stacked_power/np.log(2)
    elif stack_method == "mean":
        stacked_power = np.mean(power_matrix, axis=0)
    else:
        raise ValueError(f"stack_method must be 'median' or 'mean', got '{stack_method}'")

    # Apply correct one-sided PSD normalization
    psd_arr = onesided_psd_from_power(stacked_power, ref_df)

    # Suppress out-of-band bins so they don't contribute to the MF sum
    if f_low_hz is not None or f_high_hz is not None:
        sentinel = float(np.max(psd_arr[np.isfinite(psd_arr)])) * 1e6
        mask = np.ones(ref_N, dtype=bool)
        if f_low_hz is not None:
            mask &= f_baseband_ref >= f_low_hz
        if f_high_hz is not None:
            mask &= f_baseband_ref <= f_high_hz
        psd_arr = np.where(mask, psd_arr, sentinel)

    # Safety floor and NaN cleanup
    psd_arr = np.maximum(psd_arr, np.finfo(np.float64).tiny)
    finite = np.isfinite(psd_arr)
    if not np.all(finite):
        fill = float(np.median(psd_arr[finite])) if np.any(finite) else 1.0
        psd_arr = np.where(finite, psd_arr, fill)
        log.warning("PSD: filled %d non-finite bins with median %.3e",
                    int(np.sum(~finite)), fill)

    psd_fd = FrequencySeries(psd_arr.astype(np.float64), delta_f=float(ref_df))
    n_used = len(power_rows)

    log.info("PSD: %d scans used, %d skipped, delta_f=%.4f Hz, method=%s",
             n_used, len(skipped), ref_df, stack_method)

    meta = {
        "n_scans_used":    n_used,
        "n_scans_skipped": len(skipped),
        "delta_f_hz":      ref_df,
        "skipped_files":   skipped,
        "per_scan_power":  power_matrix if save_per_scan else None,
    }

    return psd_fd, f_baseband_ref, f_abs_ref, meta


# ---------------------------------------------------------------------------
# Save / Load HDF5
# ---------------------------------------------------------------------------

def save_psd_h5(
    out_path: Path,
    psd_fd: FrequencySeries,
    f_baseband_hz: np.ndarray,
    f_abs_hz: np.ndarray,
    meta: dict,
    nibble_name: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("PSD",           data=np.array(psd_fd).astype(np.float64))
        f.create_dataset("f_baseband_hz", data=f_baseband_hz.astype(np.float64))
        f.create_dataset("f_abs_hz",      data=f_abs_hz.astype(np.float64))

        if meta.get("per_scan_power") is not None:
            f.create_dataset(
                "PSD_per_scan",
                data=meta["per_scan_power"].astype(np.float64),
                compression="gzip", compression_opts=4,
            )

        f.attrs["delta_f_hz"]      = float(meta["delta_f_hz"])
        f.attrs["n_scans_used"]    = int(meta["n_scans_used"])
        f.attrs["n_scans_skipped"] = int(meta["n_scans_skipped"])
        f.attrs["nibble_name"]     = nibble_name
        f.attrs["psd_formula"]     = (
            "S_n(f_k) = 2 * df * median(|V_flat(f_k)|^2) for interior bins. "
            "DC and Nyquist not doubled. pyCBC matched_filter_core compatible."
        )

    log.info("Saved PSD -> %s", out_path)


def load_psd_h5(path: Path) -> tuple[FrequencySeries, np.ndarray, np.ndarray, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PSD file not found: {path}")

    with h5py.File(path, "r") as f:
        psd_arr   = f["PSD"][:].astype(np.float64)
        f_bb      = f["f_baseband_hz"][:].astype(np.float64)
        f_abs     = f["f_abs_hz"][:].astype(np.float64)
        delta_f   = float(f.attrs["delta_f_hz"])
        n_used    = int(f.attrs.get("n_scans_used", 0))
        n_skipped = int(f.attrs.get("n_scans_skipped", 0))
        per_scan  = f["PSD_per_scan"][:] if "PSD_per_scan" in f else None

    psd_fd = FrequencySeries(psd_arr, delta_f=delta_f)
    meta = {
        "n_scans_used":    n_used,
        "n_scans_skipped": n_skipped,
        "delta_f_hz":      delta_f,
        "per_scan_power":  per_scan,
    }
    return psd_fd, f_bb, f_abs, meta


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_psd(
    psd_fd: FrequencySeries,
    f_baseband_hz: np.ndarray,
    integration_time_s: Optional[float] = None,
    delta_f_hz: Optional[float] = None,
    plot: bool = True,
    save_plot: Optional[Path] = None,
) -> dict:
    psd_arr = np.asarray(psd_fd, dtype=np.float64)
    finite  = np.isfinite(psd_arr) & (psd_arr > 0)
    psd_fin = psd_arr[finite]

    stats = {
        "psd_median":     float(np.median(psd_fin)),
        "psd_min":        float(np.min(psd_fin)),
        "psd_max":        float(np.max(psd_fin)),
        "expected_floor": None,
        "excess_ratio":   None,
    }

    if integration_time_s is not None and delta_f_hz is not None:
        # For receiver-normalised data: unit variance per bin =>
        # S_n = 2 * df * 1 = 2 * df per bin
        # But radiometer fluctuation gives std ~ 1/sqrt(t_int*df),
        # so per-bin power ~ 1/(t_int*df) and S_n ~ 2/(t_int)
        expected = 2.0 * float(delta_f_hz) / (float(integration_time_s) * float(delta_f_hz))
        # simplifies to 2/t_int, but keep explicit for clarity
        expected = 2.0 / float(integration_time_s)
        stats["expected_floor"] = expected
        stats["excess_ratio"]   = stats["psd_median"] / expected
        level = "OK" if stats["excess_ratio"] < 3.0 else "WARNING"
        log.info("PSD validation [%s]: median/expected = %.2f", level, stats["excess_ratio"])

    if plot or save_plot:
        import matplotlib.pyplot as plt
        f_khz = f_baseband_hz / 1e3
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("PSD validation", fontsize=12)

        for ax, yscale in zip(axes, ("linear", "log")):
            ax.plot(f_khz, psd_arr, lw=0.8, label="Stacked PSD")
            if stats["expected_floor"] is not None:
                ax.axhline(stats["expected_floor"], ls="--", lw=1, color="C1",
                           label=f"Expected floor ({stats['expected_floor']:.2e})")
            ax.set_xlabel("Baseband frequency [kHz]")
            ax.set_ylabel("S_n(f)  [V_flat^2 * Hz]")
            ax.set_yscale(yscale)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plot:
            fig.savefig(save_plot, dpi=150, bbox_inches="tight")
        if plot:
            plt.show()
        plt.close(fig)

    return stats


# ---------------------------------------------------------------------------
# Grid alignment helper (used by matched_filter_core)
# ---------------------------------------------------------------------------

def ensure_psd_on_grid(
    psd_fd: FrequencySeries,
    delta_f_hz: float,
    n_freq: int,
    psd_floor: float = 1e-60,
) -> FrequencySeries:
    """
    Interpolate the PSD onto a target frequency grid if needed.
    Applies a safety floor before and after interpolation.
    """
    from pycbc.psd import interpolate as pycbc_interp

    psd_arr = np.asarray(psd_fd, dtype=np.float64)
    psd_arr = np.where(np.isfinite(psd_arr), psd_arr, psd_floor)
    psd_arr = np.maximum(psd_arr, psd_floor)
    psd_safe = FrequencySeries(psd_arr, delta_f=float(psd_fd.delta_f))

    target_df = float(delta_f_hz)
    if not np.isclose(float(psd_fd.delta_f), target_df, rtol=1e-6) or len(psd_fd) != n_freq:
        psd_safe = pycbc_interp(psd_safe, target_df, n_freq)

    out_arr = np.maximum(np.asarray(psd_safe, dtype=np.float64), psd_floor)
    return FrequencySeries(out_arr, delta_f=target_df)


# ---------------------------------------------------------------------------
# Per-scan PSD (diagnostics / single-scan fallback)
# ---------------------------------------------------------------------------

def psd_from_single_scan(
    v_flat: np.ndarray,
    delta_f_hz: float,
) -> FrequencySeries:
    power = np.abs(np.asarray(v_flat, dtype=np.complex128)) ** 2
    psd_arr = onesided_psd_from_power(power, delta_f_hz)
    return FrequencySeries(psd_arr.astype(np.float64), delta_f=float(delta_f_hz))


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_psd_computation(
    run_definition_path: str = "run1b_definitions.yaml",
    nibble_name: str = "nibble5",
    prepared_dir: str = "hr_data/prepared",
    output_path: Optional[str] = None,
    stack_method: str = "median",
    f_low_hz: Optional[float] = None,
    f_high_hz: Optional[float] = None,
    save_per_scan: bool = False,
    validate: bool = True,
    plot: bool = False,
    save_plot: Optional[str] = None,
) -> tuple[FrequencySeries, np.ndarray, np.ndarray]:
    cfg = RunConfig.from_yaml(run_definition_path)
    nibble_cfg = cfg.nibble(nibble_name)

    psd_fd, f_bb, f_abs, meta = compute_psd_from_prepared_dir(
        prepared_dir=prepared_dir,
        stack_method=stack_method,
        f_low_hz=f_low_hz,
        f_high_hz=f_high_hz,
        save_per_scan=save_per_scan,
    )

    if output_path is None:
        file_prefix = nibble_cfg["file_prefix"]
        out_h5 = Path(prepared_dir) / f"{file_prefix}_psd.h5"
    else:
        out_h5 = Path(output_path)

    save_psd_h5(out_h5, psd_fd, f_bb, f_abs, meta, nibble_name=nibble_name)

    if validate:
        stats = validate_psd(
            psd_fd, f_bb,
            plot=plot,
            save_plot=Path(save_plot) if save_plot else None,
        )
        log.info(
            "PSD stats: median=%.3e  min=%.3e  max=%.3e  n_used=%d",
            stats["psd_median"], stats["psd_min"], stats["psd_max"],
            meta["n_scans_used"],
        )

    return psd_fd, f_bb, f_abs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute stacked noise PSD for ADMX matched filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--run_definition", default="run1b_definitions.yaml")
    p.add_argument("-n", "--nibble_name",    default="nibble5")
    p.add_argument("--prepared_dir",         default="hr_data/prepared")
    p.add_argument("--output_path",          default=None)
    p.add_argument("--stack_method",         default="median", choices=["median", "mean"])
    p.add_argument("--f_low_hz",  type=float, default=None)
    p.add_argument("--f_high_hz", type=float, default=None)
    p.add_argument("--save_per_scan", action="store_true")
    p.add_argument("--no_validate",   action="store_true")
    p.add_argument("--plot",          action="store_true")
    p.add_argument("--save_plot",     default=None)
    p.add_argument("--log_level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    psd_fd, f_bb, f_abs = run_psd_computation(
        run_definition_path=args.run_definition,
        nibble_name=args.nibble_name,
        prepared_dir=args.prepared_dir,
        output_path=args.output_path,
        stack_method=args.stack_method,
        f_low_hz=args.f_low_hz,
        f_high_hz=args.f_high_hz,
        save_per_scan=args.save_per_scan,
        validate=not args.no_validate,
        plot=args.plot,
        save_plot=args.save_plot,
    )
    p = np.array(psd_fd)
    print(f"PSD computed: N={len(p)}, delta_f={float(psd_fd.delta_f):.6f} Hz")
    print(f"  min={np.min(p):.3e}  median={np.median(p):.3e}  max={np.max(p):.3e}")


if __name__ == "__main__":
    main()