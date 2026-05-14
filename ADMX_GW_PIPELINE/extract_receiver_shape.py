# """
# extract_receiver_shape.py
# --------------------------
# Standalone tool to build, inspect, and validate the ADMX receiver shape
# (voltage transfer function) for a given nibble.

# Relationship to binned4.py
# --------------------------
# ``binned4.run_pipeline`` builds the receiver template automatically as part
# of the main processing pipeline and caches it to an HDF5 file.  This module
# provides:
# """

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt

from admx_db_datatypes import ComplexVoltageSeries
from binned4 import (
    ReceiverTemplate,
    build_receiver_template,
    read_dat_file,
    polyphase_fft,
    MODE_NATIVE,
    MODE_PHASE_PRESERVING,
)
from config_file_handling import (
    RunConfig,
    find_parameter_row,
    parse_filename_timestamp,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build + save
# ---------------------------------------------------------------------------
def build_and_save_receiver(
    run_definition_path: str | Path,
    nibble_name: str,
    hr_root: str | Path,
    param_date: str,
    resolution_mode: str = MODE_PHASE_PRESERVING,
    seg_len: int = 64,
    crop_seconds: float = 2.0,
    window: str = "hann",
    smoothing_method: str = "savgol",
    poly_deg: int = 5,
    savgol_window_frac: float = 0.05,
    channel: int = 1,
    output_path: Optional[Path] = None,
) -> ReceiverTemplate:
    cfg = RunConfig.from_yaml(run_definition_path)
    nibble_cfg = cfg.nibble(nibble_name)
    param_df = cfg.load_parameter_df(nibble_name, param_date)
    dat_files = cfg.dat_files(nibble_name, hr_root=hr_root, channel=channel)

    if not dat_files:
        raise FileNotFoundError(
            f"No .dat files found for nibble '{nibble_name}' under '{hr_root}'."
        )

    seg = seg_len if resolution_mode == MODE_PHASE_PRESERVING else 1

    log.info(
        "Building receiver template: nibble=%s, n_files=%d, seg_len=%d, "
        "smoothing=%s, poly_deg=%d",
        nibble_name, len(dat_files), seg, smoothing_method, poly_deg,
    )

    receiver = build_receiver_template(
        dat_files=dat_files,
        param_df=param_df,
        seg_len=seg,
        crop_seconds=crop_seconds,
        window=window,
        smoothing_method=smoothing_method,
        poly_deg=poly_deg,
        savgol_window_frac=savgol_window_frac,
    )

    if output_path is None:
        file_prefix = nibble_cfg["file_prefix"]
        output_path = Path(hr_root) / f"{file_prefix}_receiver_template.h5"

    receiver.save(output_path)
    log.info("Receiver template saved → %s", output_path)
    return receiver


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_receiver(
    receiver: ReceiverTemplate,
    nibble_name: str = "",
    plot: bool = True,
    save_plot: Optional[Path] = None,
) -> dict:
    G = receiver.voltage_divider
    f = receiver.f_baseband

    warnings: list[str] = []
    errors: list[str] = []
    stats: dict = {}

    # -- Check 1: finite values -------------------------------------------
    if not np.all(np.isfinite(G)):
        errors.append(
            f"Voltage divider contains {np.sum(~np.isfinite(G))} NaN/Inf values."
        )

    # -- Check 2: positivity -----------------------------------------------
    n_nonpos = np.sum(G <= 0)
    if n_nonpos > 0:
        errors.append(
            f"Voltage divider has {n_nonpos} non-positive bins. "
            "Division will produce Inf or NaN."
        )

    # -- Check 3: dynamic range --------------------------------------------
    if np.any(G > 0):
        G_pos = G[G > 0]
        dyn_range = float(G_pos.max() / G_pos.min())
        stats["dynamic_range"] = dyn_range
        if dyn_range > 10.0:
            warnings.append(
                f"Dynamic range of voltage divider is {dyn_range:.1f}×. "
                "Expected ~1.15-1.30× for Run 1B warm electronics. "
                "Check smoothing parameters or bandpass edge handling."
            )
        elif dyn_range < 1.01:
            warnings.append(
                f"Dynamic range is only {dyn_range:.4f}× — template may be "
                "over-smoothed (nearly flat). Reduce poly_deg or widen SG window."
            )
        stats["G_min"] = float(G_pos.min())
        stats["G_max"] = float(G_pos.max())
        stats["G_mean"] = float(G_pos.mean())

    # -- Check 4: smoothness (second derivative) ---------------------------
    if len(G) > 4:
        d2 = np.diff(G, n=2)
        smoothness_ratio = float(np.std(d2) / np.mean(np.abs(G)))
        stats["smoothness_ratio"] = smoothness_ratio
        if smoothness_ratio > 0.01:
            warnings.append(
                f"Second-derivative smoothness ratio = {smoothness_ratio:.4f} "
                "(> 0.01). Template may have residual scan-to-scan noise. "
                "Try increasing savgol_window_frac."
            )

    # -- Check 5: frequency axis -------------------------------------------
    if len(f) > 1:
        if not np.all(np.diff(f) > 0):
            errors.append("Frequency axis is not monotonically increasing.")
        if f[0] < -1.0:
            errors.append(f"Frequency axis starts at {f[0]:.2f} Hz — expected >= 0 Hz.")

    stats["n_bins"] = int(len(G))
    stats["delta_f_hz"] = float(receiver.delta_f)
    stats["n_scans_averaged"] = int(receiver.n_scans)
    stats["seg_len"] = int(receiver.seg_len)
    stats["smoothing_method"] = receiver.smoothing_method

    passed = len(errors) == 0
    result = {"passed": passed, "warnings": warnings, "errors": errors, "stats": stats}

    # -- Log results -------------------------------------------------------
    if errors:
        for e in errors:
            log.error("RECEIVER VALIDATION ERROR: %s", e)
    if warnings:
        for w in warnings:
            log.warning("RECEIVER VALIDATION WARNING: %s", w)
    if passed and not warnings:
        log.info("Receiver validation passed with no issues.")

    # -- Plot --------------------------------------------------------------
    if plot or save_plot:
        _plot_receiver(receiver, nibble_name, stats, save_plot, show=plot)

    return result


def _plot_receiver(
    receiver: ReceiverTemplate,
    nibble_name: str,
    stats: dict,
    save_path: Optional[Path],
    show: bool,
) -> None:
    """Four-panel diagnostic figure for the receiver template."""
    f_khz = receiver.f_baseband / 1e3
    G = receiver.voltage_divider
    G_norm = G / G.mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"Receiver template — {nibble_name}  "
        f"({stats.get('n_scans_averaged', '?')} scans, "
        f"seg_len={stats.get('seg_len', '?')}, "
        f"method={stats.get('smoothing_method', '?')})",
        fontsize=12,
    )

    # Panel 1: voltage divider
    ax = axes[0, 0]
    ax.plot(f_khz, G, lw=1.2)
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("G(f) — voltage divider [a.u.]")
    ax.set_title("Voltage divider (√mean power)")
    ax.grid(True, alpha=0.3)

    # Panel 2: normalised shape
    ax = axes[0, 1]
    ax.plot(f_khz, G_norm, lw=1.2, color="C1")
    ax.axhline(1.0, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("G(f) / mean(G)")
    ax.set_title(
        f"Normalised shape  "
        f"(dynamic range = {stats.get('dynamic_range', float('nan')):.3f}×)"
    )
    ax.grid(True, alpha=0.3)

    # Panel 3: fractional variation (shows bandpass edges)
    ax = axes[1, 0]
    frac_var = (G - G.mean()) / G.mean()
    ax.plot(f_khz, frac_var * 100, lw=1.2, color="C2")
    ax.axhline(0, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("Baseband frequency [kHz]")
    ax.set_ylabel("Fractional deviation [%]")
    ax.set_title("Fractional deviation from mean")
    ax.grid(True, alpha=0.3)

    # Panel 4: second derivative (smoothness check)
    ax = axes[1, 1]
    if len(G) > 2:
        d2 = np.diff(G, n=2)
        f_d2 = f_khz[1:-1]
        ax.plot(f_d2, d2, lw=0.8, color="C3", alpha=0.8)
        ax.axhline(0, ls="--", lw=0.8, color="gray")
        ax.set_xlabel("Baseband frequency [kHz]")
        ax.set_ylabel("d²G/df² [a.u.]")
        ax.set_title(
            f"Second derivative  "
            f"(smoothness ratio = {stats.get('smoothness_ratio', float('nan')):.5f})"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info("Saved receiver validation plot → %s", save_path)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-scan residual diagnostic
# ---------------------------------------------------------------------------
def check_scan_residuals(
    dat_files: list[Path],
    param_df,
    receiver: ReceiverTemplate,
    n_scans: int = 10,
    seg_len: int = 64,
    crop_seconds: float = 2.0,
    window: str = "hann",
    plot: bool = True,
    save_plot: Optional[Path] = None,
) -> dict:
    indices = np.linspace(0, len(dat_files) - 1, min(n_scans, len(dat_files)),
                          dtype=int)
    sampled = [dat_files[i] for i in indices]

    residual_powers: list[np.ndarray] = []

    for path in sampled:
        row = find_parameter_row(param_df, path.name)
        if row is None:
            continue
        try:
            raw, delta_t = read_dat_file(path)
        except Exception as exc:
            log.warning("check_scan_residuals: skipping %s: %s", path.name, exc)
            continue

        import pycbc.types
        ts = pycbc.types.TimeSeries(raw.astype(np.float64), delta_t)
        if crop_seconds > 0:
            try:
                ts = ts.crop(crop_seconds, crop_seconds)
            except Exception:
                pass
        x = np.asarray(ts, dtype=np.float64)

        if seg_len == 1:
            import pyfftw
            fft_v = pyfftw.interfaces.numpy_fft.rfft(x)
            freqs = np.fft.rfftfreq(len(x), d=delta_t)
        else:
            fft_v, freqs, _ = polyphase_fft(x, delta_t, seg_len, window)

        # apply receiver
        fft_flat = receiver.apply(fft_v, freqs)
        power_flat = np.abs(fft_flat) ** 2

        # normalise to mean = 1 (same as ADMX's step before subtracting 1)
        mean_p = np.mean(power_flat)
        if mean_p > 0:
            residual_powers.append(power_flat / mean_p)

    if not residual_powers:
        raise RuntimeError("No scans could be processed for residual check.")

    stack = np.vstack(residual_powers)
    mean_res = np.mean(stack, axis=0)
    std_res = np.std(stack, axis=0)

    # check: mean should be close to 1 everywhere
    max_dev = float(np.max(np.abs(mean_res - 1.0)))
    if max_dev > 0.05:
        log.warning(
            "check_scan_residuals: max fractional deviation from unity = %.4f "
            "(> 5%%). Receiver shape may not be fully captured.", max_dev
        )
    else:
        log.info(
            "check_scan_residuals: max deviation from unity = %.4f (< 5%% — good).",
            max_dev,
        )

    if plot or save_plot:
        f_khz = freqs / 1e3
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Post-flatten residuals  ({len(residual_powers)} scans sampled)",
            fontsize=12,
        )

        ax = axes[0]
        ax.plot(f_khz, mean_res, lw=1.2, label="mean")
        ax.fill_between(f_khz, mean_res - std_res, mean_res + std_res,
                        alpha=0.3, label="±1σ")
        ax.axhline(1.0, ls="--", lw=0.8, color="gray")
        ax.set_xlabel("Baseband frequency [kHz]")
        ax.set_ylabel("Normalised power (mean=1)")
        ax.set_title("Residual power after receiver division")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(f_khz, (mean_res - 1.0) * 100, lw=1.2, color="C1")
        ax.axhline(0, ls="--", lw=0.8, color="gray")
        ax.set_xlabel("Baseband frequency [kHz]")
        ax.set_ylabel("(mean/1 − 1) [%]")
        ax.set_title("Fractional residual from unity")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plot is not None:
            fig.savefig(save_plot, dpi=150, bbox_inches="tight")
            log.info("Saved residual plot → %s", save_plot)
        if plot:
            plt.show()
        plt.close(fig)

    return {
        "mean_residuals": mean_res,
        "std_residuals": std_res,
        "f_baseband_hz": freqs,
        "n_scans_checked": len(residual_powers),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and validate the ADMX receiver shape template.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--run_definition", default="run1b_definitions.yaml")
    p.add_argument("-n", "--nibble_name", default="nibble5")
    p.add_argument("--hr_root", default="hr_data")
    p.add_argument("--param_date", default="2018_05_19")
    p.add_argument("--channel", type=int, default=1)

    p.add_argument(
        "--resolution_mode",
        choices=[MODE_NATIVE, MODE_PHASE_PRESERVING],
        default=MODE_PHASE_PRESERVING,
    )
    p.add_argument("--seg_len", type=int, default=64)
    p.add_argument("--crop_seconds", type=float, default=2.0)
    p.add_argument("--window", default="hann",
                   choices=["hann", "blackman", "hamming", "none"])
    p.add_argument("--smoothing_method", default="savgol",
                   choices=["savgol", "poly"])
    p.add_argument("--poly_deg", type=int, default=5)
    p.add_argument("--savgol_window_frac", type=float, default=0.05)

    p.add_argument("--output_path", default=None,
                   help="Override default output path for receiver HDF5.")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip validation plots.")
    p.add_argument("--save_plot", default=None,
                   help="Save validation plot to this path.")
    p.add_argument("--check_residuals", action="store_true",
                   help="Also check per-scan residuals after flattening.")
    p.add_argument("--n_residual_scans", type=int, default=10,
                   help="Number of scans to sample for residual check.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_path = Path(args.output_path) if args.output_path else None

    receiver = build_and_save_receiver(
        run_definition_path=args.run_definition,
        nibble_name=args.nibble_name,
        hr_root=args.hr_root,
        param_date=args.param_date,
        resolution_mode=args.resolution_mode,
        seg_len=args.seg_len,
        crop_seconds=args.crop_seconds,
        window=args.window,
        smoothing_method=args.smoothing_method,
        poly_deg=args.poly_deg,
        savgol_window_frac=args.savgol_window_frac,
        channel=args.channel,
        output_path=out_path,
    )

    print(receiver)

    save_plot = Path(args.save_plot) if args.save_plot else None
    result = validate_receiver(
        receiver,
        nibble_name=args.nibble_name,
        plot=not args.no_plot,
        save_plot=save_plot,
    )

    if result["errors"]:
        print("\nVALIDATION ERRORS:")
        for e in result["errors"]:
            print(f"  ERROR: {e}")
    if result["warnings"]:
        print("\nVALIDATION WARNINGS:")
        for w in result["warnings"]:
            print(f"  WARNING: {w}")
    if result["passed"]:
        print("\nValidation passed.")
    print("\nStats:", result["stats"])

    if args.check_residuals:
        cfg = RunConfig.from_yaml(args.run_definition)
        param_df = cfg.load_parameter_df(args.nibble_name, args.param_date)
        dat_files = cfg.dat_files(
            args.nibble_name, hr_root=args.hr_root, channel=args.channel
        )
        save_resid = (
            Path(args.save_plot).with_suffix("").as_posix() + "_residuals.png"
            if args.save_plot else None
        )
        check_scan_residuals(
            dat_files=dat_files,
            param_df=param_df,
            receiver=receiver,
            n_scans=args.n_residual_scans,
            seg_len=args.seg_len if args.resolution_mode == MODE_PHASE_PRESERVING else 1,
            crop_seconds=args.crop_seconds,
            window=args.window,
            plot=not args.no_plot,
            save_plot=Path(save_resid) if save_resid else None,
        )


if __name__ == "__main__":
    main()