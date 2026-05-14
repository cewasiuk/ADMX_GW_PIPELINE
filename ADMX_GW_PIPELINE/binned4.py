# """
# binned4.py
# ----------
# FFT, frequency-resolution reduction, receiver-shape extraction, and
# receiver-flattening for ADMX High-Resolution .dat files.

# Resolution modes (toggled via ``resolution_mode`` argument)
# ------------------------------------------------------------
# """

from __future__ import annotations

import argparse
import array
import json
import logging
import struct
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pycbc.types
import pyfftw

from admx_db_datatypes import ComplexVoltageSeries, ScanParameters
from config_file_handling import (
    RunConfig,
    find_parameter_row,
    PARAMETER_COLUMNS,
)

log = logging.getLogger(__name__)
pyfftw.interfaces.cache.enable()

# ---------------------------------------------------------------------------
# Resolution mode constants
# ---------------------------------------------------------------------------
MODE_NATIVE = "native"
MODE_PHASE_PRESERVING = "phase_preserving"
_VALID_MODES = {MODE_NATIVE, MODE_PHASE_PRESERVING}


# ---------------------------------------------------------------------------
# Low-level .dat file reader
# ---------------------------------------------------------------------------
def read_dat_file(path: Path) -> tuple[np.ndarray, float]:
    with open(path, "rb") as fh:
        header_size = struct.unpack("q", fh.read(8))[0]
        fh.read(header_size)
        h1_size = struct.unpack("q", fh.read(8))[0]
        h1 = json.loads(fh.read(h1_size))
        delta_t = float(h1["x_spacing"]) * 1e-6      # us -> s
        npts = struct.unpack("Q", fh.read(8))[0]
        buf = array.array("f")
        buf.frombytes(fh.read(npts * 4))
    return np.asarray(buf, dtype=np.float32), delta_t


# ---------------------------------------------------------------------------
# Phase-preserving polyphase channeliser
# ---------------------------------------------------------------------------
def polyphase_fft(
    x: np.ndarray,
    delta_t: float,
    seg_len: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, float]:

    if seg_len < 1:
        raise ValueError(f"seg_len must be >= 1, got {seg_len}")
    if not (seg_len & (seg_len - 1) == 0):
        log.warning("seg_len=%d is not a power of 2; FFT will be slower.", seg_len)

    N = len(x)
    M = N // seg_len
    if M < 1:
        raise ValueError(
            f"seg_len={seg_len} > N_samples={N}; reduce seg_len or crop less."
        )

    window = window.lower()
    if window == "hann":
        win = np.hanning(seg_len).astype(np.float64)
    elif window == "blackman":
        win = np.blackman(seg_len).astype(np.float64)
    elif window == "hamming":
        win = np.hamming(seg_len).astype(np.float64)
    elif window in ("none", "rectangular", "boxcar"):
        win = np.ones(seg_len, dtype=np.float64)
    else:
        raise ValueError(f"Unknown window '{window}'.")

    win_correction = seg_len / win.sum()
    n_out = seg_len // 2 + 1
    fft_sum = np.zeros(n_out, dtype=np.complex128)

    for m in range(M):
        seg = x[m * seg_len : (m + 1) * seg_len].astype(np.float64)
        fft_sum += pyfftw.interfaces.numpy_fft.rfft(seg * win)

    fft_avg = (fft_sum / M) * win_correction
    freqs = np.fft.rfftfreq(seg_len, d=delta_t)
    delta_f_out = 1.0 / (seg_len * delta_t)

    return fft_avg, freqs, delta_f_out


# ---------------------------------------------------------------------------
# Per-scan processing
# ---------------------------------------------------------------------------
def process_scan(
    path: Path,
    param_df,
    receiver=None,          # no longer used; kept for API compatibility
    b_field_tesla: float = 0.0,
    resolution_mode: str = MODE_PHASE_PRESERVING,
    seg_len: int = 64,
    crop_seconds: float = 2.0,
    window: str = "hann",
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Optional[tuple]:

    import gc

    if resolution_mode not in _VALID_MODES:
        raise ValueError(f"resolution_mode must be one of {_VALID_MODES}")

    row = find_parameter_row(param_df, path.name)
    if row is None:
        log.debug("process_scan: skipping %s (no param row)", path.name)
        return None

    stem = path.stem
    if output_dir is not None:
        out_path = Path(output_dir) / f"{stem}_binned.h5"
        if out_path.exists() and not overwrite:
            log.debug("process_scan: skipping %s (output exists)", path.name)
            return None

    try:
        raw, delta_t = read_dat_file(path)
    except Exception as exc:
        log.warning("process_scan: failed to read %s: %s", path.name, exc)
        return None

    fstart_hz = float(row["Start_Frequency"]) * 1e6
    fstop_hz  = float(row["Stop_Frequency"])  * 1e6

    # ── Crop transients ───────────────────────────────────────────────────
    ts = pycbc.types.TimeSeries(raw, delta_t)   # raw is already float32
    del raw
    if crop_seconds > 0:
        try:
            ts = ts.crop(crop_seconds, crop_seconds)
        except Exception:
            pass
    x      = np.asarray(ts, dtype=np.float32)  # ~40 MB for 100 s scan
    del ts
    N_time = len(x)
    delta_f_native = 1.0 / (N_time * delta_t)

    # ── Polyphase path ────────────────────────────────────────────────────
    if resolution_mode == MODE_PHASE_PRESERVING and seg_len > 1:
        fft_v, freqs, delta_f_out = polyphase_fft(x, delta_t, seg_len, window)
        del x; gc.collect()
        N_bins       = len(fft_v)
        freqs_abs    = freqs + fstart_hz
        seg_len_used = seg_len

        # Polyphase output is a single coherent spectrum — safe to decimate
        # directly for the polynomial receiver fit.
        D        = max(1, N_bins // 2000)
        pwr_dec  = np.abs(fft_v[::D]) ** 2
        f_nd     = np.linspace(0.0, 1.0, len(pwr_dec))
        poly_obj = np.poly1d(np.polyfit(f_nd, pwr_dec, 8))
        del pwr_dec, f_nd; gc.collect()

        CHUNK           = 500_000
        voltage_divider = np.empty(N_bins, dtype=np.float32)
        for s in range(0, N_bins, CHUNK):
            e   = min(s + CHUNK, N_bins)
            fn  = np.linspace(s / N_bins, e / N_bins, e - s)
            div = np.sqrt(np.maximum(poly_obj(fn), 1e-30)).astype(np.float32)
            fft_v[s:e]           /= div
            voltage_divider[s:e]  = div

        fft_flat                = fft_v
        receiver_divider_interp = voltage_divider

    # ── Native path (chunked FFT for memory efficiency) ───────────────────
    else:
        seg_len_used = 1
        delta_f_out  = delta_f_native
        N_bins       = N_time // 2 + 1
        freqs        = np.fft.rfftfreq(N_time, d=delta_t)
        freqs_abs    = freqs + fstart_hz

        # ── Receiver shape: ADMX procedure (full FFT, actual frequency axis) ─
        # Replicate exactly what the ADMX analysis code does:
        #   1. FFT the full time series at full resolution
        #   2. Compute |FFT|^2 (power spectrum)
        #   3. Fit 8th-order polynomial using the actual Hz frequency axis
        #
        # Using the actual frequency axis (not normalised [0,1]) matches the
        # ADMX procedure and avoids the Runge oscillations we saw with
        # normalisation + decimation. The full-resolution FFT is computed
        # here only for the receiver fit and immediately discarded — it is
        # not stored. The chunked FFT below stores the data at full resolution.
        fft_recv   = np.fft.rfft(x.astype(np.float64))   # float64 for fit accuracy
        freq_axis  = np.fft.rfftfreq(N_time, d=delta_t)   # actual Hz axis
        pow_freq   = np.abs(fft_recv) ** 2
        del fft_recv
        poly_obj   = np.poly1d(np.polyfit(freq_axis, pow_freq, 8))
        del pow_freq
        gc.collect()

        # ── Chunked FFT: full-resolution storage only ─────────────────────
        # The chunking here is purely for memory management — each chunk is
        # ~8 MB of complex64, keeping peak RAM well under 500 MB.

        NCHUNK     = 10
        chunk_bins = N_bins // NCHUNK
        f_starts   = [ci * chunk_bins for ci in range(NCHUNK)]
        f_ends     = f_starts[1:] + [N_bins]

        # rfft of L samples gives L//2+1 bins, so L = (n_bins - 1) * 2
        t_lens    = [(fe - fs - 1) * 2 for fs, fe in zip(f_starts, f_ends)]
        t_starts  = [0] + list(np.cumsum(t_lens[:-1]))
        t_ends_l  = [ts2 + tl for ts2, tl in zip(t_starts, t_lens)]
        t_ends_l[-1] = N_time   # last chunk gets all remaining samples

        fft_flat        = np.empty(N_bins, dtype=np.complex64)
        voltage_divider = np.empty(N_bins, dtype=np.float32)

        for ci in range(NCHUNK):
            fs, fe   = f_starts[ci], f_ends[ci]
            ts2, te2 = t_starts[ci], t_ends_l[ci]
            seg      = x[ts2:te2].astype(np.float32)
            fchunk   = np.fft.rfft(seg).astype(np.complex64)
            del seg
            n_out = fe - fs
            fft_flat[fs:fe] = fchunk[:n_out]
            del fchunk

        del x; gc.collect()

        # ── Normalise in-place using the whole-file polynomial ────────────
        CHUNK = 500_000
        for s in range(0, N_bins, CHUNK):
            e      = min(s + CHUNK, N_bins)
            f_eval = freq_axis[s:e]   # actual Hz values for these bins
            div    = np.sqrt(np.maximum(poly_obj(f_eval), 1e-30)).astype(np.float32)
            fft_flat[s:e]           /= div
            voltage_divider[s:e]     = div

        receiver_divider_interp = voltage_divider

    # ── Truncate to exact native bin count (fixes off-by-one from chunking)
    N_exact         = N_time // 2 + 1
    fft_flat        = fft_flat[:N_exact]
    freqs           = freqs[:N_exact]
    freqs_abs       = freqs_abs[:N_exact]
    voltage_divider = voltage_divider[:N_exact]
    receiver_divider_interp = voltage_divider

    # ── Package result ────────────────────────────────────────────────────
    scan_params = ScanParameters.from_dataframe_row(row, b_field_tesla=b_field_tesla)

    cvs = ComplexVoltageSeries(
        yvalues=fft_flat,
        f_baseband_start_hz=float(freqs[0]),
        f_baseband_stop_hz=float(freqs[-1]) + delta_f_out,
        f_abs_start_hz=fstart_hz,
        scan_params=scan_params,
        delta_f_hz=delta_f_out,
    )

    if output_dir is not None:
        out_path = Path(output_dir) / f"{stem}_binned.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _save_scan_h5(
            out_path=out_path,
            fft_raw=fft_flat,
            fft_flat=fft_flat,
            receiver_divider=receiver_divider_interp,
            freqs_baseband=freqs,
            freqs_abs=freqs_abs,
            delta_t=delta_t,
            delta_f_native=delta_f_native,
            delta_f_out=delta_f_out,
            N_time=N_time,
            seg_len=seg_len_used,
            resolution_mode=resolution_mode,
            fstart_abs_hz=fstart_hz,
            fstop_abs_hz=fstop_hz,
            row=row,
        )
        log.info("Saved -> %s", out_path)

    # Return cvs and receiver shape so the caller can inspect/plot both
    return cvs, receiver_divider_interp, freqs


# ---------------------------------------------------------------------------
# HDF5 output
# ---------------------------------------------------------------------------
def _save_scan_h5(
    out_path: Path,
    fft_raw: np.ndarray,
    fft_flat: np.ndarray,
    receiver_divider: np.ndarray,
    freqs_baseband: np.ndarray,
    freqs_abs: np.ndarray,
    delta_t: float,
    delta_f_native: float,
    delta_f_out: float,
    N_time: int,
    seg_len: int,
    resolution_mode: str,
    fstart_abs_hz: float,
    fstop_abs_hz: float,
    row,
) -> None:
    with h5py.File(out_path, "w") as f:
        f.create_dataset("FFT_Binned",     data=fft_raw.astype(np.complex128))
        f.create_dataset("FFT_BinnedFlat", data=fft_flat.astype(np.complex128))
        f.create_dataset("Receiver_Shape", data=receiver_divider.astype(np.float64))
        f.create_dataset("f_baseband",     data=freqs_baseband.astype(np.float64))
        f.create_dataset("f_abs",          data=freqs_abs.astype(np.float64))

        f.attrs["delta_t"]         = float(delta_t)
        f.attrs["delta_f_native"]  = float(delta_f_native)
        f.attrs["delta_f_out"]     = float(delta_f_out)
        f.attrs["delta_f_binned"]  = float(delta_f_out)   # legacy key
        f.attrs["seg_len"]         = int(seg_len)
        f.attrs["N_time"]          = int(N_time)
        f.attrs["num_bins"]        = int(len(fft_flat))
        f.attrs["bin_factor"]      = int(seg_len)
        f.attrs["fstart_abs_hz"]   = float(fstart_abs_hz)
        f.attrs["fstop_abs_hz"]    = float(fstop_abs_hz)
        f.attrs["resolution_mode"] = resolution_mode
        f.attrs["notes"] = (
            "FFT_BinnedFlat: complex spectrum divided by Receiver_Shape. "
            "Receiver_Shape = sqrt(8th-order poly fit to decimated whole-file "
            "|FFT|^2). Phase is preserved. "
            f"resolution_mode={resolution_mode}, seg_len={seg_len}."
        )

        g = f.require_group("run_params")
        for col in PARAMETER_COLUMNS:
            key = col.rstrip("?")
            val = row.get(col, row.get(key, None))
            if val is None:
                continue
            try:
                g.attrs[key] = float(val)
            except (TypeError, ValueError):
                g.attrs[key] = np.bytes_(str(val))


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def run_pipeline(
    run_definition_path: str = "run1b_definitions.yaml",
    nibble_name: str = "nibble5",
    hr_root: str = "hr_data",
    param_date: str = "2018_05_19",
    output_subfolder: str = "binned_hr_data",
    resolution_mode: str = MODE_PHASE_PRESERVING,
    seg_len: int = 64,
    crop_seconds: float = 2.0,
    window: str = "hann",
    channel: int = 1,
    overwrite: bool = False,
) -> list[Path]:
    cfg       = RunConfig.from_yaml(run_definition_path)
    b_field   = cfg.b_field(nibble_name)
    param_df  = cfg.load_parameter_df(nibble_name, param_date)
    dat_files = cfg.dat_files(nibble_name, hr_root=hr_root, channel=channel)

    if not dat_files:
        raise FileNotFoundError(
            f"No .dat files found for nibble '{nibble_name}' under '{hr_root}'."
        )

    output_dir = Path(hr_root) / output_subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    n_total = len(dat_files)

    for i, dat_path in enumerate(dat_files):
        log.info("[%d/%d] Processing %s", i + 1, n_total, dat_path.name)

        from config_file_handling import parse_filename_timestamp
        ts = parse_filename_timestamp(dat_path.name)
        if ts is not None:
            cut, reason = cfg.is_scan_cut(nibble_name, ts)
            if cut:
                log.info("  Skipping (cut: %s)", reason)
                continue

        result = process_scan(
            path=dat_path,
            param_df=param_df,
            b_field_tesla=b_field,
            resolution_mode=resolution_mode,
            seg_len=seg_len,
            crop_seconds=crop_seconds,
            window=window,
            output_dir=output_dir,
            overwrite=overwrite,
        )

        if result is not None:
            out_path = output_dir / f"{dat_path.stem}_binned.h5"
            if out_path.exists():
                outputs.append(out_path)

    log.info("Done. %d / %d scans written to %s", len(outputs), n_total, output_dir)
    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADMX HR data FFT + receiver-flatten pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-r", "--run_definition", default="run1b_definitions.yaml")
    p.add_argument("-n", "--nibble_name",    default="nibble5")
    p.add_argument("--hr_root",              default="hr_data")
    p.add_argument("--param_date",           default="2018_05_19")
    p.add_argument("--output_subfolder",     default="binned_hr_data")
    p.add_argument("--channel",  type=int,   default=1)
    p.add_argument(
        "--resolution_mode",
        choices=[MODE_NATIVE, MODE_PHASE_PRESERVING],
        default=MODE_PHASE_PRESERVING,
    )
    p.add_argument("--seg_len",      type=int,   default=64)
    p.add_argument("--crop_seconds", type=float, default=2.0)
    p.add_argument(
        "--window", default="hann",
        choices=["hann", "blackman", "hamming", "none"],
    )
    p.add_argument("--overwrite",  action="store_true")
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
    outputs = run_pipeline(
        run_definition_path=args.run_definition,
        nibble_name=args.nibble_name,
        hr_root=args.hr_root,
        param_date=args.param_date,
        output_subfolder=args.output_subfolder,
        resolution_mode=args.resolution_mode,
        seg_len=args.seg_len,
        crop_seconds=args.crop_seconds,
        window=args.window,
        channel=args.channel,
        overwrite=args.overwrite,
    )
    print(f"Processed {len(outputs)} scans.")


if __name__ == "__main__":
    main()

